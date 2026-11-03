"""Name a missing capability, try a small experiment, and keep the procedure that worked."""

from __future__ import annotations

import json
import re

from genesis.capabilities.service import ModuleService
from genesis.core.configuration.settings import Settings
from genesis.core.model_interface.service import ModelService
from genesis.storage.database import Database, Skill

MAX_NAMED = 8
GAP_MAX_TOKENS = 1536
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

GAP_RETRY = (
    "That reply was not the required JSON object. "
    'Return only {"missing":[],"known":[],'
    '"experiments":[{"name":"read-folder","hypothesis":"The directory named in the objective holds the module.",'
    '"probe":"Inspect that directory, install the module folder, and invoke it with the stated task.",'
    '"expect":"Installed","uses":["inspect","install","invoke"]}]}. '
    "Do not ask the user."
)
_PLACEHOLDERS = {"what to try", "text that proves success", "why this trial", "..."}


class LearningError(ValueError):
    pass


class ExperimentSpec:
    def __init__(
        self,
        name: str,
        hypothesis: str,
        probe: str,
        expect: str,
        uses: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.hypothesis = hypothesis
        self.probe = probe
        self.expect = expect
        self.uses = uses


class GapAssessment:
    def __init__(
        self,
        missing: tuple[str, ...],
        known: tuple[str, ...],
        experiments: tuple[ExperimentSpec, ...],
    ) -> None:
        self.missing = missing
        self.known = known
        self.experiments = experiments


class LearningService:
    def __init__(self, settings: Settings, db: Database, models: ModelService, modules: ModuleService) -> None:
        self.settings = settings
        self.db = db
        self.models = models
        self.modules = modules

    def assess(self, request: str) -> GapAssessment:
        matched = matching_skills(request, self.db.list_skills())
        if matched:
            return GapAssessment((), tuple(skill.name for skill in matched), ())
        limit = self.settings.learning.max_experiments
        messages: list[dict[str, str]] = [
            {"role": "system", "content": gap_system(limit)},
            {"role": "user", "content": gap_request(request, self.modules.inventory())},
        ]
        reply = self._ask(messages)
        try:
            return self._ground(parse_gap(reply, max_experiments=limit))
        except LearningError:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": GAP_RETRY})
            second = self._ask(messages)
            try:
                return self._ground(parse_gap(second, max_experiments=limit))
            except LearningError as exc:
                raise LearningError(f"{exc} Model reply: {second[:500]}") from exc

    def _ground(self, assessment: GapAssessment) -> GapAssessment:
        return ground_gap(assessment, action_names(self.modules.inventory()))

    def context_for(self, request: str, objective_id: str) -> str:
        parts: list[str] = []
        gaps = self.db.list_gaps(objective_id)
        if gaps:
            missing = _names(gaps[-1].missing_json)
            known = _names(gaps[-1].known_json)
            if missing:
                parts.append("Missing capabilities: " + ", ".join(missing) + ".")
            if known:
                parts.append("Known skills: " + ", ".join(known) + ".")
        for skill in matching_skills(request, self.db.list_skills()):
            parts.append(f"Skill {skill.name}:\n{skill.body}")
        for experiment in self.db.list_experiments(objective_id):
            state = "succeeded" if experiment.success else "failed"
            parts.append(
                f"Experiment {experiment.name} {state}. "
                f"Expected {experiment.expected}. Outcome: {experiment.outcome}"
            )
        return "\n\n".join(parts)[:4000]

    def _ask(self, messages: list[dict[str, str]]) -> str:
        limit = max(self.settings.model.max_tokens, GAP_MAX_TOKENS)
        return self.models.generate(
            messages,
            temperature=0.0,
            max_tokens=limit,
            response_format={"type": "json_object"},
        ).content


def covered_gap_reply() -> str:
    return json.dumps({"missing": [], "known": [], "experiments": []})


def gap_system(max_experiments: int) -> str:
    return (
        "You compare an objective with the actions Genesis already has. "
        "Return only a JSON object with missing, known, and experiments. "
        "One shape that uses actions already in the list: "
        '{"missing":[],"known":[],"experiments":[{"name":"read-folder",'
        '"hypothesis":"The directory named in the objective holds the module.",'
        '"probe":"Inspect that directory, install the module folder, and invoke it with the stated task.",'
        '"expect":"Installed","uses":["inspect","install","invoke"]}]}. '
        "known is empty when there are no saved skills. Never copy action names into known or missing. "
        "missing lists only capabilities that are not actions. "
        "When the actions can carry out the objective, missing is empty and experiments say how to use them. "
        "A module that is already inside this project is found with inspect, installed with install, and called with invoke. "
        "uses names actions from the list you are given. An experiment without those names is ignored. "
        f"Use at most {max_experiments} experiments. Keep each field to one sentence. "
        "name is letters, numbers, hyphen, or underscore. "
        "Do not carry out the objective."
    )


def gap_request(request: str, inventory: str) -> str:
    actions, skills = inventory_brief(inventory)
    return (
        "The objective is quoted data. Name the trials that would carry it out. "
        "Do not repeat the action list inside known or missing. "
        "Write a real probe and a real expect. "
        'Return only {"missing":[],"known":[],'
        '"experiments":[{"name":"read-folder",'
        '"hypothesis":"The directory named in the objective holds the module.",'
        '"probe":"Inspect that directory, install the module folder, and invoke it with the stated task.",'
        '"expect":"Installed","uses":["inspect","install","invoke"]}]}.\n\n'
        f"Actions: {actions or 'none'}\n"
        f"Saved skills: {skills or 'none'}\n\n"
        f"Objective:\n{request}"
    )


def inventory_brief(inventory: str) -> tuple[str, str]:
    actions: list[str] = []
    skills: list[str] = []
    section = "actions"
    for line in inventory.splitlines():
        if line.strip() == "Skills:":
            section = "skills"
            continue
        if not line.startswith("- "):
            continue
        token = line[2:].split(" ", 1)[0].strip()
        if not token:
            continue
        target = skills if section == "skills" else actions
        if token not in target:
            target.append(token)
    return ", ".join(actions), ", ".join(skills)


def gap_message(assessment: GapAssessment) -> str:
    parts: list[str] = []
    if assessment.missing:
        parts.append("Missing: " + ", ".join(assessment.missing) + ".")
    if assessment.known:
        parts.append("Covered by " + ", ".join(assessment.known) + ".")
    return " ".join(parts) or "No missing capability."


def procedure_text(spec: ExperimentSpec, outcome: str) -> str:
    return (
        f"{spec.hypothesis}\n\n"
        f"Probe: {spec.probe}\n"
        f"Expected: {spec.expect}\n"
        f"Result: {outcome.strip()}"
    )


def matching_skills(request: str, skills: list[Skill]) -> list[Skill]:
    words = {word.lower() for word in _WORD.findall(request)}
    return [skill for skill in skills if skill.name.lower() in words]


def parse_gap(text: str, *, max_experiments: int) -> GapAssessment:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", visible, flags=re.DOTALL)
    if match is None:
        raise LearningError("Learning assessment did not return a JSON object.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LearningError("Learning assessment returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise LearningError("Learning assessment did not return a JSON object.")
    missing = _string_list(payload.get("missing"), "missing")
    known = _string_list(payload.get("known"), "known")
    raw = payload.get("experiments", [])
    if not isinstance(raw, list):
        raise LearningError("Learning assessment returned experiments that are not a list.")
    experiments: list[ExperimentSpec] = []
    seen: set[str] = set()
    for item in raw:
        if len(experiments) >= max_experiments:
            break
        if not isinstance(item, dict):
            raise LearningError("Learning assessment returned an experiment that is not an object.")
        name = str(item.get("name") or "").strip()
        hypothesis = str(item.get("hypothesis") or "").strip()
        probe = str(item.get("probe") or "").strip()
        expect = str(item.get("expect") or "").strip()
        uses = _string_list(item.get("uses"), f"uses for {name or 'experiment'}")
        if _NAME.fullmatch(name) is None:
            raise LearningError(f"Experiment name '{name}' is not a skill name.")
        if not hypothesis or not probe or not expect:
            raise LearningError(f"Experiment '{name}' needs a hypothesis, probe, and expect.")
        if probe.casefold() in _PLACEHOLDERS or expect.casefold() in _PLACEHOLDERS:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        experiments.append(ExperimentSpec(name, hypothesis, probe, expect, uses))
    return GapAssessment(tuple(missing), tuple(known), tuple(experiments))


def action_names(inventory: str) -> set[str]:
    names: set[str] = set()
    for line in inventory.splitlines():
        if not line.startswith("- "):
            continue
        token = line[2:].split(" ", 1)[0].strip()
        if token:
            names.add(token.casefold())
    return names


def ground_gap(assessment: GapAssessment, allowed: set[str]) -> GapAssessment:
    missing = tuple(item for item in assessment.missing if item.casefold() not in allowed)
    experiments = tuple(
        spec
        for spec in assessment.experiments
        if spec.uses and all(name.casefold() in allowed for name in spec.uses)
    )
    return GapAssessment(missing, assessment.known, experiments)


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise LearningError(f"Learning assessment returned {label} that is not a list.")
    names: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in names:
            names.append(text)
        if len(names) >= MAX_NAMED:
            break
    return tuple(names)


def _names(encoded: str) -> list[str]:
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
