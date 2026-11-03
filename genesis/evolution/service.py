"""Keep every tested generation and choose a parent from more than one measurement."""

from __future__ import annotations

import time
from pathlib import Path

from genesis.core.configuration.settings import Settings
from genesis.development.service import DevelopmentService, ModificationError
from genesis.storage.database import Candidate, Database, Evaluation, Generation, Proposal
from genesis.tools.research.actions import ResearchAction


class ArchiveError(ValueError):
    pass


class ArchiveService:
    def __init__(self, settings: Settings, db: Database, development: DevelopmentService) -> None:
        self.settings = settings
        self.db = db
        self.development = development

    def baseline(self) -> Generation:
        for generation in self.db.list_generations():
            if generation.status == "baseline":
                return generation
        return self.db.save_generation(None, None, "live", "baseline")

    def propose(
        self,
        objective_id: str | None,
        path: str,
        rationale: str,
        old: str,
        new: str,
        parent: str | None = None,
    ) -> tuple[Proposal, Candidate, Generation]:
        parent_generation = self.baseline() if parent is None or not parent.strip() else self._resolve(parent)
        source = self._source(parent_generation)
        proposal, candidate = self.development.propose(objective_id, path, rationale, old, new, source)
        generation = self.db.save_generation(candidate.id, parent_generation.id, candidate.id[:8], "open")
        return proposal, candidate, generation

    def evaluate(self, target: str, benchmark: str | None = None) -> str:
        name = self._benchmark(benchmark)
        generation = self._resolve(target)
        success, elapsed, report = self._measure(generation)
        if success == 0 and generation.status == "open":
            generation = self.db.set_generation_status(generation.id, "rejected")
        evaluation = self.db.save_evaluation(generation.id, name, success, elapsed, report[-4000:])
        return _score_line(generation, evaluation)

    def retain(self, target: str) -> str:
        generation = self._find(target)
        if generation is None or generation.candidate_id is None:
            if generation is not None and generation.candidate_id is None:
                raise ArchiveError("The live baseline is already kept.")
            candidate = self.development.retain(target)
            return f"Retained {candidate.id}. The live tree was not replaced."
        candidate = self.development.retain(generation.candidate_id)
        stored = self.db.set_generation_status(generation.id, "retained")
        return f"Retained generation {stored.id} for candidate {candidate.id}. The live tree was not replaced."

    def select(self, benchmark: str | None = None) -> str:
        name = self._benchmark(benchmark)
        ranked: list[tuple[Generation, Evaluation]] = []
        for generation in self.db.list_generations():
            latest = self.db.latest_evaluation(generation.id, name)
            if latest is not None:
                ranked.append((generation, latest))
        if not ranked:
            raise ArchiveError(f"No generation has been evaluated on {name}.")
        ranked.sort(
            key=lambda item: (
                -item[1].success,
                float("inf") if item[1].elapsed_ms is None else item[1].elapsed_ms,
                item[0].created_at,
                item[0].id,
            )
        )
        winner, evaluation = ranked[0]
        lines = [
            f"Benchmark: {name}",
            _score_line(winner, evaluation).replace("Generation", "Selected", 1),
            "Considered:",
        ]
        for generation, item in ranked:
            lines.append(_score_line(generation, item))
        lines.append("Other generations stay in the archive. The live tree was not replaced.")
        reason = "\n".join(lines)
        self.db.save_selection(name, winner.id, reason)
        return reason

    def lineage(self) -> str:
        rows = self.db.list_generations()
        if not rows:
            self.baseline()
            rows = self.db.list_generations()
        lines = []
        benchmark = self._benchmark(None)
        for item in rows:
            latest = self.db.latest_evaluation(item.id, benchmark)
            score = "not evaluated" if latest is None else _score_line(item, latest)
            parent = item.parent_id or "-"
            lines.append(f"{item.id} {item.status} parent {parent} {item.label} {score}")
        chosen = self.db.latest_selection(benchmark)
        if chosen is not None:
            lines.append(f"Selected parent: {chosen.generation_id}")
        return "\n".join(lines)

    def revisit(self, target: str, path: str) -> str:
        generation = self._resolve(target)
        return self.development.inspect(path, self._source(generation))

    def execute(self, objective_id: str | None, action: ResearchAction) -> str:
        try:
            if action.name == "propose":
                _proposal, candidate, generation = self.propose(
                    objective_id,
                    action.path,
                    action.rationale or action.task or action.query,
                    action.old,
                    action.text,
                    action.parent or None,
                )
                return f"Generation {generation.id} parent {generation.parent_id}\n{self.development.compare(candidate.id)}"
            if action.name == "retain":
                return self.retain(action.package or action.query)
            if action.name == "evaluate":
                return self.evaluate(action.package or action.query or "live", action.task or None)
            if action.name == "select":
                target = action.package or action.query
                if target and self._find(target) is not None:
                    benchmark = self._benchmark(None)
                    baseline = self.baseline()
                    if self.db.latest_evaluation(baseline.id, benchmark) is None:
                        self.evaluate(baseline.id)
                    self.evaluate(target)
                    return self.select(None)
                return self.select(target or None)
            if action.name == "lineage":
                return self.lineage()
            if action.name == "revisit":
                return self.revisit(action.package or action.query, action.path)
        except (ArchiveError, ModificationError) as exc:
            return f"Archive failed: {exc}"
        except KeyError:
            return "Archive failed: that generation does not exist."
        return f"Archive failed: unknown action '{action.name}'."

    def _benchmark(self, benchmark: str | None) -> str:
        name = (benchmark or self.settings.evolution.benchmark).strip()
        return name or "tests"

    def _resolve(self, target: str) -> Generation:
        generation = self._find(target)
        if generation is None:
            raise ArchiveError(f"{target.strip() or 'live'} does not exist.")
        return generation

    def _find(self, target: str) -> Generation | None:
        text = target.strip()
        if not text or text in {"live", "baseline"}:
            return self.baseline()
        generation = self.db.get_generation(text)
        if generation is not None:
            return generation
        return self.db.get_generation_for_candidate(text)

    def _source(self, generation: Generation) -> Path | None:
        if generation.candidate_id is None:
            return None
        candidate = self.db.get_candidate(generation.candidate_id)
        if candidate is None:
            raise ArchiveError(f"Candidate {generation.candidate_id} does not exist.")
        return Path(candidate.directory)

    def _measure(self, generation: Generation) -> tuple[float, float, str]:
        directory = self.development.root if generation.candidate_id is None else self._source(generation)
        if directory is None:
            raise ArchiveError(f"{generation.id} has no source tree.")
        started = time.perf_counter()
        try:
            result = self.development.runner(
                directory,
                self.development.root,
                self.settings.development.test_timeout_seconds,
            )
        except Exception as exc:
            result = (1, str(exc))
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        code = int(result[0])
        output = str(result[1])
        if len(result) > 2 and result[2] is not None:
            elapsed = float(result[2])
        if generation.candidate_id:
            candidate = self.db.get_candidate(generation.candidate_id)
            if candidate is not None and candidate.status != "retained":
                status = "passed" if code == 0 else "failed"
                self.db.finish_candidate(candidate.id, status, output[-4000:], elapsed)
        return (1.0 if code == 0 else 0.0), elapsed, output


def _score_line(generation: Generation, evaluation: Evaluation) -> str:
    elapsed = "unknown" if evaluation.elapsed_ms is None else f"{evaluation.elapsed_ms:.1f}"
    return (
        f"Generation {generation.id} benchmark {evaluation.benchmark} "
        f"success {evaluation.success:.1f} elapsed {elapsed}"
    )
