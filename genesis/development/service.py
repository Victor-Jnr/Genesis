"""Propose a source change, test it in a separate copy, and keep the live tree recoverable."""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

from genesis.core.configuration.settings import Settings
from genesis.storage.database import Candidate, Database, Proposal
from genesis.tools.research.actions import ResearchAction

COPY_DIRECTORIES = ("genesis", "tests", "config", "scripts")
COPY_FILES = (
    "pytest.ini",
    "requirements.txt",
    "requirements-dev.txt",
    "README.md",
    "CHANGELOG.md",
)
ALLOWED_PREFIXES = ("genesis/", "tests/", "config/")
ALLOWED_FILES = {
    "README.md",
    "CHANGELOG.md",
    "pytest.ini",
    "requirements.txt",
    "requirements-dev.txt",
}
ALLOWED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".ini", ".txt"}
MAX_FILE_CHARS = 200_000
MAX_EDIT_CHARS = 20_000


class ModificationError(ValueError):
    pass


class DevelopmentService:
    def __init__(self, settings: Settings, db: Database, root: Path, runner=None) -> None:
        self.settings = settings
        self.db = db
        self.root = root.resolve()
        self.candidates = settings.resolve_path(settings.development.candidates_directory, root)
        self.runner = runner or run_candidate_tests

    def inspect(self, raw: str) -> str:
        path = self._allowed(raw)
        if path.is_dir():
            lines = [
                item.relative_to(self.root).as_posix()
                for item in sorted(path.rglob("*"))
                if item.is_file() and _is_listed_file(item.relative_to(self.root).as_posix())
            ]
            return "\n".join(lines[:200]) or f"{raw} is empty."
        text = _read(path)
        if len(text) > 12000:
            return text[:12000] + "\n... truncated"
        return text

    def propose(
        self,
        objective_id: str | None,
        raw_path: str,
        rationale: str,
        old: str,
        new: str,
    ) -> tuple[Proposal, Candidate]:
        reason = rationale.strip()
        previous = old
        replacement = new
        if not reason:
            raise ModificationError("A proposal needs a rationale.")
        if not previous or not replacement:
            raise ModificationError("A proposal needs the old text and the replacement.")
        if len(previous) > MAX_EDIT_CHARS or len(replacement) > MAX_EDIT_CHARS:
            raise ModificationError("The edited text is too long.")
        if previous == replacement:
            raise ModificationError("The replacement is the same as the original text.")
        path = self._allowed(raw_path)
        if path.is_dir():
            raise ModificationError("Choose a file, not a directory.")
        relative = path.relative_to(self.root).as_posix()
        original = _read(path)
        count = original.count(previous)
        if count != 1:
            raise ModificationError(f"The text to replace occurs {count} times.")
        updated = original.replace(previous, replacement, 1)
        if relative.endswith(".py"):
            try:
                ast.parse(updated)
            except SyntaxError as exc:
                raise ModificationError(f"The candidate is not valid Python: {exc.msg}") from exc
        proposal = self.db.save_proposal(
            objective_id,
            relative,
            reason,
            previous,
            replacement,
            hashlib.sha256(original.encode("utf-8")).hexdigest(),
        )
        candidate = self._materialize(proposal, original, updated)
        return proposal, candidate

    def test(self, candidate_id: str) -> Candidate:
        candidate = self._require(candidate_id)
        if candidate.status == "retained":
            raise ModificationError(f"{candidate_id} is already retained.")
        started = time.perf_counter()
        try:
            code, output = self.runner(
                Path(candidate.directory),
                self.root,
                self.settings.development.test_timeout_seconds,
            )
        except Exception as exc:
            output = str(exc)
            code = 1
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        status = "passed" if code == 0 else "failed"
        return self.db.finish_candidate(candidate.id, status, output[-4000:], elapsed)

    def compare(self, candidate_id: str) -> str:
        candidate = self._require(candidate_id)
        proposal = self._proposal(candidate.proposal_id)
        live = _read(self.root / proposal.path)
        candidate_file = Path(candidate.directory) / proposal.path
        changed = _read(candidate_file) if candidate_file.is_file() else ""
        live_hash = hashlib.sha256(live.encode("utf-8")).hexdigest()
        lines = [
            f"Parent revision: {candidate.parent_revision}",
            f"Candidate {candidate.id}: {candidate.status}",
            f"File: {proposal.path}",
            f"Rationale: {proposal.rationale}",
            f"Live file matches the previous version: {live_hash == proposal.base_sha256}",
            f"Candidate contains the replacement: {proposal.new_text in changed}",
            f"Elapsed ms: {candidate.elapsed_ms if candidate.elapsed_ms is not None else 'not run'}",
            "Report:",
            candidate.report or "not tested",
        ]
        return "\n".join(lines)

    def retain(self, candidate_id: str) -> Candidate:
        candidate = self._require(candidate_id)
        if candidate.status != "passed":
            raise ModificationError(f"Only a passing candidate can be retained. Status is {candidate.status}.")
        return self.db.finish_candidate(candidate.id, "retained", candidate.report, candidate.elapsed_ms)

    def archive(self) -> str:
        rows = self.db.list_candidates()
        if not rows:
            return "No candidates."
        return "\n".join(
            f"{item.id} {item.status} parent {item.parent_revision} {self._proposal(item.proposal_id).path}"
            for item in rows
        )

    def execute(self, objective_id: str | None, action: ResearchAction) -> str:
        try:
            if action.name == "inspect":
                return self.inspect(action.path)
            if action.name == "propose":
                _proposal, candidate = self.propose(
                    objective_id,
                    action.path,
                    action.rationale or action.task or action.query,
                    action.old,
                    action.text,
                )
                return self.compare(candidate.id)
            if action.name == "test_candidate":
                candidate = self.test(action.package or action.query)
                return self.compare(candidate.id)
            if action.name == "compare":
                return self.compare(action.package or action.query)
            if action.name == "retain":
                candidate = self.retain(action.package or action.query)
                return f"Retained {candidate.id}. The live tree was not replaced."
            if action.name == "archive":
                return self.archive()
        except ModificationError as exc:
            return f"Modification failed: {exc}"
        except KeyError:
            return "Modification failed: that candidate does not exist."
        return f"Modification failed: unknown action '{action.name}'."

    def _materialize(self, proposal: Proposal, original: str, updated: str) -> Candidate:
        destination = self.candidates / proposal.id
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        _copy_source(self.root, destination)
        target = destination / proposal.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
        live = _read(self.root / proposal.path)
        if live != original:
            raise ModificationError("The live file changed while the candidate was written.")
        return self.db.save_candidate(proposal.id, _parent_revision(self.root), str(destination))

    def _allowed(self, raw: str) -> Path:
        text = raw.strip().replace("\\", "/")
        if not text or text.startswith("/") or ":" in text:
            raise ModificationError("Use a path inside the project.")
        relative = Path(text)
        if any(part == ".." for part in relative.parts):
            raise ModificationError("The path leaves the project.")
        path = (self.root / relative).resolve()
        try:
            visible = path.relative_to(self.root)
        except ValueError as exc:
            raise ModificationError("The path leaves the project.") from exc
        posix = visible.as_posix()
        if path.is_dir():
            if posix not in {"genesis", "tests", "config"} and not posix.startswith(ALLOWED_PREFIXES):
                raise ModificationError(f"{posix} cannot be inspected.")
            return path
        if not _is_listed_file(posix):
            raise ModificationError(f"{posix} cannot be modified.")
        if not path.is_file():
            raise ModificationError(f"{posix} does not exist.")
        return path

    def _require(self, candidate_id: str) -> Candidate:
        candidate = self.db.get_candidate(candidate_id)
        if candidate is None:
            raise ModificationError(f"{candidate_id} does not exist.")
        return candidate

    def _proposal(self, proposal_id: str) -> Proposal:
        proposal = self.db.get_proposal(proposal_id)
        if proposal is None:
            raise ModificationError(f"Proposal {proposal_id} does not exist.")
        return proposal


def run_candidate_tests(candidate: Path, repo: Path, timeout: float) -> tuple[int, str]:
    link_project_venv(repo, candidate)
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    python = candidate / ".venv" / folder / name
    if not python.exists():
        raise ModificationError(f"The project virtual environment is not available at {python}.")
    completed = subprocess.run(
        [str(python), "-m", "pytest", "-q", "--tb=line"],
        cwd=candidate,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return completed.returncode, output


def link_project_venv(repo: Path, candidate: Path) -> None:
    target = (repo / ".venv").resolve()
    link = candidate / ".venv"
    if not target.is_dir():
        raise ModificationError("The project virtual environment does not exist.")
    if link.exists():
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ModificationError(completed.stderr.strip() or "Could not link the project virtual environment.")
        return
    link.symlink_to(target, target_is_directory=True)


def _copy_source(root: Path, destination: Path) -> None:
    for name in COPY_DIRECTORIES:
        source = root / name
        if source.is_dir():
            shutil.copytree(
                source,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for name in COPY_FILES:
        source = root / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _is_listed_file(posix: str) -> bool:
    if posix in ALLOWED_FILES:
        return True
    if not posix.startswith(ALLOWED_PREFIXES):
        return False
    return Path(posix).suffix in ALLOWED_SUFFIXES


def _read(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ModificationError(f"{path.name} is not a text file.") from exc
    if len(text) > MAX_FILE_CHARS:
        raise ModificationError(f"{path.name} is too large to modify.")
    return text


def _parent_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unversioned"
    if completed.returncode != 0:
        return "unversioned"
    return completed.stdout.strip() or "unversioned"
