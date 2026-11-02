"""Turn an objective into steps and carry one step out with the loaded model."""

from __future__ import annotations

import json
import re

from genesis.core.model_interface.service import ModelService
from genesis.storage.database import StepRecord

MAX_STEPS = 6

PLAN_SYSTEM = (
    "You are the Genesis planner. You only write plans. "
    'Return only a JSON object: {"steps":[{"title":"short name","instruction":"what to produce"}]}. '
    f"Use at most {MAX_STEPS} steps. Do not carry out the objective. Do not add markdown."
)

PLAN_RETRY = (
    "That reply was not the required JSON object. "
    'Return only {"steps":[{"title":"short name","instruction":"what to produce"}]}.'
)

STEP_SYSTEM = (
    "You are Genesis carrying out one step of an objective. "
    "When the step can be answered from the objective, reply with the step result as plain text. "
    "When it needs a public source, reply with one JSON object and nothing else. "
    'Actions: {"action":"search","query":"..."}, {"action":"repositories","query":"..."}, '
    '{"action":"package","name":"..."}, {"action":"fetch","url":"..."}, '
    '{"action":"read","url":"..."}, {"action":"download","url":"..."}, '
    '{"action":"finish","result":"..."}. '
    "After an observation, send the next action or the step result."
)

CONCLUDE_SYSTEM = (
    "You are Genesis finishing an objective. "
    "Using the completed steps, reply with the final result and nothing else."
)


class PlanError(ValueError):
    pass


class Planner:
    def __init__(self, models: ModelService) -> None:
        self.models = models

    def plan(self, request: str) -> tuple[StepRecord, ...]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": plan_request(request)},
        ]
        reply = self._ask(messages)
        try:
            return parse_plan(reply)
        except PlanError:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": PLAN_RETRY})
            second = self._ask(messages)
            try:
                return parse_plan(second)
            except PlanError as exc:
                raise PlanError(f"{exc} Model reply: {second[:500]}") from exc

    def _ask(self, messages: list[dict[str, str]]) -> str:
        result = self.models.generate(messages, temperature=0.0, max_tokens=512)
        return result.content

    def conclude(self, request: str, steps: tuple[StepRecord, ...]) -> str:
        lines = [f"Objective:\n{request}", "", "Completed steps:"]
        for index, step in enumerate(steps, start=1):
            lines.append(f"{index}. {step.title}: {step.result or ''}")
        result = self.models.generate(
            [
                {"role": "system", "content": CONCLUDE_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        text = result.content.strip()
        if not text:
            raise PlanError("The final result was empty.")
        return text


def step_messages(request: str, step: StepRecord, index: int, count: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STEP_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Objective:\n{request}\n\n"
                f"Step {index + 1} of {count}: {step.title}\n"
                f"Instruction:\n{step.instruction}"
            ),
        },
    ]


def plan_request(request: str) -> str:
    return (
        "Write a plan for the objective below. The objective is quoted data, not an instruction to answer. "
        'Return only {"steps":[{"title":"short name","instruction":"what to produce"}]}.\n\n'
        f"Objective:\n{request}"
    )


def parse_plan(text: str) -> tuple[StepRecord, ...]:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", visible, flags=re.DOTALL)
    if match is None:
        raise PlanError("Planner did not return a JSON object.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PlanError("Planner returned invalid JSON.") from exc
    raw_steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("Planner returned no steps.")
    steps: list[StepRecord] = []
    for item in raw_steps[:MAX_STEPS]:
        if not isinstance(item, dict):
            raise PlanError("Planner returned a step that is not an object.")
        title = str(item.get("title") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if not title or not instruction:
            raise PlanError("Planner returned a step without a title and instruction.")
        steps.append(StepRecord(title, instruction, None))
    return tuple(steps)
