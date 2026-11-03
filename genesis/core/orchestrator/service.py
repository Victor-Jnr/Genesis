"""Run objectives apart from the chat that created them."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from genesis.capabilities.manifest import ModuleError
from genesis.capabilities.service import ModuleService
from genesis.development.service import DevelopmentService
from genesis.evolution.service import ArchiveService
from genesis.cognition.learning import (
    LearningError,
    LearningService,
    gap_message,
    procedure_text,
)
from genesis.cognition.planner import PlanError, Planner, step_messages
from genesis.tools.execution.service import ExecutionService
from genesis.tools.research.actions import parse_action
from genesis.tools.research.service import ResearchService
from genesis.core.configuration.settings import Settings
from genesis.core.model_interface.llama_cpp_backend import ModelLoadError
from genesis.core.model_interface.service import ModelNotReady, ModelService
from genesis.monitoring.logging import log_event
from genesis.storage.database import Database, Objective, ObjectiveEvent, StepRecord

log = logging.getLogger("genesis")

PENDING = "pending"
PLANNING = "planning"
EXECUTING = "executing"
WAITING = "waiting"
PAUSED = "paused"
COMPLETED = "completed"
FAILED = "failed"


class UnknownObjective(KeyError):
    pass


class ObjectiveStateError(RuntimeError):
    pass


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        models: ModelService,
        research: ResearchService,
        modules: ModuleService,
        development: DevelopmentService,
        archive: ArchiveService,
        execution: ExecutionService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.models = models
        self.research = research
        self.modules = modules
        self.development = development
        self.archive = archive
        self.execution = execution
        self.planner = Planner(models)
        self.learning = LearningService(settings, db, models, modules)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="genesis-objectives", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def notify_model(self) -> None:
        if self.models.snapshot().loaded:
            self.db.requeue_waiting_objectives()
        self._wake.set()

    def submit(self, request: str) -> Objective:
        text = request.strip()
        if not text:
            raise ValueError("Objective is empty.")
        objective = self.db.create_objective(text)
        self._event(objective.id, "created", text)
        self._wake.set()
        return objective

    def get(self, objective_id: str) -> tuple[Objective, list[ObjectiveEvent]]:
        objective = self._require(objective_id)
        return objective, self.db.list_objective_events(objective_id)

    def list_objectives(self) -> list[Objective]:
        return self.db.list_objectives()

    def pause(self, objective_id: str) -> Objective:
        objective = self._require(objective_id)
        if objective.status in {COMPLETED, FAILED}:
            raise ObjectiveStateError(f"Cannot pause an objective that is {objective.status}.")
        if objective.status != PAUSED:
            objective = self._mark(objective, status=PAUSED)
            self._event(objective.id, "paused", "Paused.")
        return objective

    def resume(self, objective_id: str) -> Objective:
        objective = self._require(objective_id)
        if objective.status == COMPLETED:
            raise ObjectiveStateError("Cannot resume a completed objective.")
        if objective.status != PENDING:
            objective = self._mark(objective, status=PENDING, error=None, protect_pause=False)
            self._event(objective.id, "resumed", "Resumed.")
        self._wake.set()
        return objective

    def run_available(self) -> None:
        while not self._stop.is_set() and self._tick():
            continue

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._tick():
                continue
            self._wake.wait(0.5)
            self._wake.clear()

    def _tick(self) -> bool:
        if self._stop.is_set():
            return False
        objective = self.db.next_runnable_objective()
        if objective is None:
            return False
        self._execute(objective)
        return True

    def _execute(self, objective: Objective) -> None:
        if self._stop.is_set():
            return
        current = self.db.get_objective(objective.id)
        if current is None or current.status in {PAUSED, COMPLETED, FAILED, WAITING}:
            return
        if not self.models.snapshot().loaded:
            detail = self.models.snapshot().detail or "No GGUF model is loaded."
            self._mark(current, status=WAITING, error=detail)
            self._event(current.id, "waiting", detail)
            return
        try:
            current = self._prepare(current)
            if current.status == PAUSED or self._stop.is_set():
                return
            while current.current_step < len(current.steps):
                if self._stop.is_set():
                    return
                fresh = self.db.get_objective(current.id)
                if fresh is None or fresh.status == PAUSED:
                    return
                index = current.current_step
                step = current.steps[index]
                self._event(current.id, "step", step.title)
                text = self._perform_step(current.id, current.request, step, index, len(current.steps))
                updated = list(current.steps)
                updated[index] = StepRecord(step.title, step.instruction, text)
                current = self._mark(
                    current,
                    steps=tuple(updated),
                    current_step=index + 1,
                    status=EXECUTING,
                )
                if current.status == PAUSED:
                    return
            if self._stop.is_set():
                return
            final = self.planner.conclude(current.request, current.steps)
            self._mark(current, status=COMPLETED, result=final, error=None, current_step=len(current.steps))
            self._event(current.id, "completed", final[:500])
        except (PlanError, LearningError) as exc:
            self._fail(current, str(exc))
        except ModelNotReady as exc:
            self._mark(current, status=WAITING, error=exc.detail)
            self._event(current.id, "waiting", exc.detail)
        except ModelLoadError as exc:
            self._fail(current, str(exc))
        except Exception as exc:
            log.exception("objective failed")
            self._fail(current, str(exc))

    def _perform_step(self, objective_id: str, request: str, step: StepRecord, index: int, count: int) -> str:
        context = self.learning.context_for(request, objective_id)
        messages = step_messages(request, step, index, count, context)
        limit = self.settings.research.max_actions
        for _ in range(limit + 1):
            reply = self.models.generate(messages, temperature=0.0, max_tokens=512).content.strip()
            action = parse_action(reply)
            if action is None or action.name == "finish":
                text = reply if action is None else action.result
                if not text:
                    raise PlanError(f"Step '{step.title}' returned an empty result.")
                return text
            if action.name in {"inventory", "install", "invoke", "remember", "skills"}:
                observation = self.modules.execute(objective_id, action)
            elif action.name in {"inspect", "test_candidate", "compare", "archive"}:
                observation = self.development.execute(objective_id, action)
            elif action.name in {"propose", "retain", "lineage", "evaluate", "select", "revisit"}:
                observation = self.archive.execute(objective_id, action)
            elif action.name == "run":
                observation = self.execution.execute(objective_id, action)
            else:
                observation = self.research.execute(objective_id, action)
            self._event(objective_id, action.name, observation)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
        raise PlanError(f"Step '{step.title}' used more than {limit} research actions.")

    def _learn(self, current: Objective) -> None:
        if self.db.list_gaps(current.id) or self.db.list_experiments(current.id):
            return
        assessment = self.learning.assess(current.request)
        self.db.save_gap(current.id, list(assessment.missing), list(assessment.known))
        self._event(current.id, "gap", gap_message(assessment))
        for spec in assessment.experiments:
            if self._stop.is_set():
                return
            fresh = self.db.get_objective(current.id)
            if fresh is None or fresh.status == PAUSED:
                return
            try:
                outcome = self._perform_step(
                    current.id,
                    current.request,
                    StepRecord(spec.name, spec.probe),
                    0,
                    1,
                )
            except PlanError as exc:
                outcome = str(exc)
            success = spec.expect.casefold() in outcome.casefold()
            self.db.save_experiment(
                current.id,
                spec.name,
                spec.hypothesis,
                spec.probe,
                spec.expect,
                outcome[:4000],
                success,
            )
            self._event(current.id, "experiment", f"{spec.name} {'succeeded' if success else 'failed'}.")
            if not success:
                continue
            try:
                self.modules.remember(spec.name, procedure_text(spec, outcome[:4000]))
            except ModuleError as exc:
                self._event(current.id, "retain-failed", str(exc))
                continue
            self._event(current.id, "retained", spec.name)

    def _prepare(self, current: Objective) -> Objective:
        if current.steps:
            return self._mark(current, status=EXECUTING, error=None)
        current = self._mark(current, status=PLANNING, error=None)
        self._event(current.id, "planning", "Planning steps.")
        if self._stop.is_set():
            return current
        self._learn(current)
        fresh = self.db.get_objective(current.id)
        if fresh is None or fresh.status == PAUSED or self._stop.is_set():
            return fresh or current
        context = self.learning.context_for(fresh.request, fresh.id)
        steps = self.planner.plan(fresh.request, context)
        fresh = self.db.get_objective(fresh.id)
        status = PAUSED if fresh is not None and fresh.status == PAUSED else EXECUTING
        current = self._mark(current, steps=steps, current_step=0, status=status)
        self._event(current.id, "planned", f"{len(steps)} steps.")
        return current

    def _fail(self, objective: Objective, message: str) -> None:
        self._mark(objective, status=FAILED, error=message)
        self._event(objective.id, "failed", message)

    def _mark(self, objective: Objective, *, protect_pause: bool = True, **changes: object) -> Objective:
        fresh = self._require(objective.id)
        if protect_pause and fresh.status == PAUSED and changes.get("status") in {PENDING, PLANNING, EXECUTING}:
            changes["status"] = PAUSED
        updated = replace(fresh, **changes)
        return self.db.save_objective(updated)

    def _require(self, objective_id: str) -> Objective:
        objective = self.db.get_objective(objective_id)
        if objective is None:
            raise UnknownObjective(objective_id)
        return objective

    def _event(self, objective_id: str, kind: str, message: str) -> None:
        self.db.add_objective_event(objective_id, kind, message)
        log_event(logging.INFO, "objective", objective_id=objective_id, kind=kind, detail=message[:500])
