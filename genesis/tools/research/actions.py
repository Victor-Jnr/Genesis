"""Recognize a research action inside a model reply."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTIONS = {
    "search",
    "repositories",
    "package",
    "fetch",
    "read",
    "download",
    "inventory",
    "install",
    "invoke",
    "remember",
    "skills",
    "inspect",
    "propose",
    "test_candidate",
    "compare",
    "retain",
    "archive",
    "lineage",
    "evaluate",
    "select",
    "revisit",
    "finish",
}


@dataclass(frozen=True)
class ResearchAction:
    name: str
    query: str = ""
    url: str = ""
    package: str = ""
    result: str = ""
    path: str = ""
    task: str = ""
    text: str = ""
    old: str = ""
    rationale: str = ""
    parent: str = ""


def parse_action(text: str) -> ResearchAction | None:
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", visible, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "action" not in payload:
        return None
    name = str(payload.get("action") or "").strip().lower()
    if name not in ACTIONS:
        return None
    return ResearchAction(
        name=name,
        query=str(payload.get("query") or "").strip(),
        url=str(payload.get("url") or "").strip(),
        package=str(payload.get("name") or payload.get("package") or "").strip(),
        result=str(payload.get("result") or "").strip(),
        path=str(payload.get("path") or "").strip(),
        task=str(payload.get("task") or "").strip(),
        text=str(payload.get("text") or payload.get("new") or "").strip(),
        old=str(payload.get("old") or "").strip(),
        rationale=str(payload.get("rationale") or "").strip(),
        parent=str(payload.get("parent") or "").strip(),
    )
