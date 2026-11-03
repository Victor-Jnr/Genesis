"""Install, remember, and call modules without editing the core application."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from genesis.capabilities.loader import call_capability, load_capability
from genesis.capabilities.manifest import Manifest, ModuleError, load_manifest
from genesis.core.configuration.settings import Settings
from genesis.core.runtime import project_venv_python
from genesis.storage.database import Database, ModuleRecord, Skill
from genesis.tools.research.actions import ResearchAction

_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$")
BUILTIN_CAPABILITIES = (
    ("search", "Search the public web"),
    ("repositories", "Search GitHub and GitLab"),
    ("package", "Read a Python package from PyPI"),
    ("fetch", "Fetch a public page"),
    ("read", "Read documentation and list install requirements"),
    ("download", "Download a public file"),
    ("browse", "Read page text without running JavaScript"),
    ("run", "Run one local program when execution is enabled"),
    ("install", "Register a module folder that is already inside the project"),
    ("invoke", "Call an installed module"),
    ("remember", "Save a procedure"),
    ("skills", "List saved procedures"),
    ("inspect", "Read a project source file"),
    ("propose", "Write a candidate change without replacing the live file"),
    ("test_candidate", "Run the tests in a candidate copy"),
    ("compare", "Compare a candidate with the live version"),
    ("retain", "Keep a passing candidate without replacing the live tree"),
    ("archive", "List candidate versions"),
    ("lineage", "List generations and the selected parent"),
    ("evaluate", "Measure a generation on the benchmark"),
    ("select", "Choose a parent by success and elapsed time"),
    ("revisit", "Read a file from an earlier generation"),
)


class ModuleService:
    def __init__(self, settings: Settings, db: Database, root: Path, runner=None) -> None:
        self.settings = settings
        self.db = db
        self.root = root.resolve()
        self.directory = settings.resolve_path(settings.modules.directory, root)
        self.skills_directory = settings.resolve_path(settings.modules.skills_directory, root)
        self.runner = runner or install_requirements

    def startup(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.skills_directory.mkdir(parents=True, exist_ok=True)
        self.discover()

    def discover(self) -> list[ModuleRecord]:
        records: list[ModuleRecord] = []
        if not self.directory.is_dir():
            return records
        for manifest_path in sorted(self.directory.glob("*/manifest.json")):
            try:
                records.append(self.install(manifest_path.parent))
            except ModuleError as exc:
                module_id = manifest_path.parent.name
                records.append(self._mark_failed(module_id, manifest_path.parent, str(exc)))
        return records

    def install_path(self, raw: str) -> ModuleRecord:
        return self.install(self._source_path(raw))

    def install(self, source: Path) -> ModuleRecord:
        origin = self._inside(source)
        if not origin.is_dir():
            raise ModuleError(f"{origin} is not a module directory.")
        manifest = load_manifest(origin / "manifest.json")
        destination = self._inside(self.directory / manifest.id)
        if origin != destination:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(origin, destination)
        existing = self.db.get_module(manifest.id)
        if existing is not None and existing.status == "active" and Path(existing.path).resolve() == destination:
            return existing
        requirements = destination / "requirements.txt"
        if _requires_install(requirements):
            try:
                self.runner(requirements)
            except Exception as exc:
                self._mark_failed(manifest.id, destination, str(exc), manifest)
                raise ModuleError(f"Installing requirements for {manifest.id} failed: {exc}") from exc
        try:
            instance = load_capability(destination, manifest.entry_point)
            _call_shutdown(instance)
        except ModuleError as exc:
            self._mark_failed(manifest.id, destination, str(exc), manifest)
            raise
        except Exception as exc:
            self._mark_failed(manifest.id, destination, str(exc), manifest)
            raise ModuleError(f"Loading {manifest.id} failed: {exc}") from exc
        return self.db.save_module(
            manifest.id,
            manifest.name,
            manifest.version,
            manifest.description,
            "active",
            str(destination),
            _manifest_json(manifest),
            None,
        )

    def invoke(self, objective_id: str | None, name: str, task: str) -> str:
        record = self.db.get_module(name)
        if record is None or record.status != "active":
            raise ModuleError(f"{name} is not an active module.")
        manifest = load_manifest(Path(record.path) / "manifest.json")
        instance = load_capability(Path(record.path), manifest.entry_point)
        text, verified = call_capability(
            instance,
            task,
            {"objective_id": objective_id or "", "module_id": manifest.id, "root": str(self.root)},
        )
        if not verified:
            raise ModuleError(f"{name} returned a result that failed verification: {text}")
        return text

    def remember(self, name: str, body: str) -> Skill:
        skill_name = name.strip()
        text = body.strip()
        if _SKILL_NAME.fullmatch(skill_name) is None:
            raise ModuleError("Skill name must be letters, numbers, hyphen, or underscore.")
        if not text:
            raise ModuleError("Skill text is empty.")
        if len(text) > 20000:
            raise ModuleError("Skill text is too long.")
        skill = self.db.save_skill(skill_name, text)
        self.skills_directory.mkdir(parents=True, exist_ok=True)
        (self.skills_directory / f"{skill_name}.md").write_text(text + "\n", encoding="utf-8")
        return skill

    def search_skills(self, query: str) -> list[Skill]:
        needle = query.strip().lower()
        if not needle:
            return self.db.list_skills()
        return [
            skill
            for skill in self.db.list_skills()
            if needle in skill.name.lower() or needle in skill.body.lower()
        ]

    def inventory(self) -> str:
        lines = ["Capabilities:"]
        for name, description in BUILTIN_CAPABILITIES:
            lines.append(f"- {name} (built-in, active) {description}")
        for module in self.db.list_modules():
            lines.append(
                f"- {module.id} {module.version} (module, {module.status}) {module.description}"
            )
        skills = self.db.list_skills()
        if skills:
            lines.append("Skills:")
            for skill in skills:
                lines.append(f"- {skill.name}")
        return "\n".join(lines)

    def execute(self, objective_id: str | None, action: ResearchAction) -> str:
        try:
            if action.name == "inventory":
                return self.inventory()
            if action.name == "install":
                record = self.install(self._source_path(action.path))
                return f"Installed {record.id} {record.version} ({record.status}). {record.description}"
            if action.name == "invoke":
                module_name = action.package or action.query
                if not module_name or not action.task:
                    raise ModuleError("invoke requires name and task.")
                return self.invoke(objective_id, module_name, action.task)
            if action.name == "remember":
                skill = self.remember(action.package or action.query, action.text or action.task)
                return f"Remembered skill {skill.name}."
            if action.name == "skills":
                found = self.search_skills(action.query)
                if not found:
                    return "No skills matched."
                return "\n".join(f"- {skill.name}: {skill.body[:240]}" for skill in found)
        except ModuleError as exc:
            return f"Module failed: {exc}"
        return f"Module failed: unknown action '{action.name}'."

    def _source_path(self, raw: str) -> Path:
        text = raw.strip()
        if not text:
            raise ModuleError("install requires a path.")
        path = Path(text)
        if not path.is_absolute():
            path = self.root / path
        return path

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ModuleError(f"{path} is outside the project.") from exc
        return resolved

    def _mark_failed(
        self,
        module_id: str,
        path: Path,
        error: str,
        manifest: Manifest | None = None,
    ) -> ModuleRecord:
        return self.db.save_module(
            module_id,
            manifest.name if manifest else module_id,
            manifest.version if manifest else "0",
            manifest.description if manifest else "",
            "failed",
            str(path),
            _manifest_json(manifest) if manifest else "{}",
            error,
        )


def requirements_command(requirements: Path) -> list[str]:
    return [str(project_venv_python()), "-m", "pip", "install", "-r", str(requirements)]


def install_requirements(requirements: Path) -> None:
    command = requirements_command(requirements)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModuleError(detail[-500:] or f"pip exited {completed.returncode}")


def _requires_install(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            return True
    return False


def _manifest_json(manifest: Manifest) -> str:
    import json

    return json.dumps(
        {
            "id": manifest.id,
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
            "source": manifest.source,
            "dependencies": list(manifest.dependencies),
            "entry_point": manifest.entry_point,
        }
    )


def _call_shutdown(instance: object) -> None:
    method = getattr(instance, "shutdown", None)
    if callable(method):
        method()
