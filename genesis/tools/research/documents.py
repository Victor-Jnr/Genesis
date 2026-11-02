"""Pull dependency and install lines out of documentation text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

_PIP = re.compile(r"(?im)^[ \t]*\$?\s*(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\s+([^\r\n#]+)")
_FENCE = re.compile(r"(?is)```requirements(?:\.txt)?\s*\n(.*?)```")
_REQ_LINE = re.compile(
    r"(?im)^[ \t]*([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?\s*(?:==|>=|<=|~=|!=|>|<)\s*\S+"
)
_PYPROJECT = re.compile(r"(?is)dependencies\s*=\s*\[(.*?)\]")
_QUOTED = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?[^"']*["']""")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*")
_FLAG = re.compile(r"^--?[A-Za-z0-9]")


@dataclass(frozen=True)
class DocumentNotes:
    dependencies: tuple[str, ...]
    install_requirements: tuple[str, ...]
    excerpt: str


def read_documentation(text: str, *, limit: int = 1500) -> DocumentNotes:
    dependencies: list[str] = []
    installs: list[str] = []
    for match in _PIP.finditer(text):
        command = " ".join(match.group(0).split()).lstrip("$").strip()
        installs.append(command)
        dependencies.extend(_packages_from_pip(match.group(1)))
        dependencies.extend(_fenced_requirements(text))
        dependencies.extend(_requirement_lines(text))
        dependencies.extend(_pyproject_dependencies(text))
    excerpt = text.strip()
    if len(excerpt) > limit:
        excerpt = excerpt[:limit] + "..."
    return DocumentNotes(tuple(_unique(dependencies)), tuple(_unique(installs)), excerpt)


def html_to_text(raw: str) -> str:
    without_blocks = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", raw)
    with_breaks = re.sub(r"(?i)<\s*/?\s*(p|div|h[1-6]|li|br|tr|section|article)\b[^>]*>", "\n", without_blocks)
    text = re.sub(r"(?s)<[^>]+>", " ", with_breaks)
    text = unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _packages_from_pip(tokens: str) -> list[str]:
    names: list[str] = []
    for token in tokens.split():
        if _FLAG.match(token) or token in {"-r", "-c", "-e", "."}:
            continue
        if token.startswith("-"):
            continue
        match = _NAME.match(token)
        if match:
            names.append(match.group(0))
    return names


def _fenced_requirements(text: str) -> list[str]:
    names: list[str] = []
    for block in _FENCE.findall(text):
        for line in block.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or stripped.startswith(("-", ".")):
                continue
            match = _NAME.match(stripped)
            if match:
                names.append(match.group(0))
    return names


def _requirement_lines(text: str) -> list[str]:
    return [match.group(1) for match in _REQ_LINE.finditer(text)]


def _pyproject_dependencies(text: str) -> list[str]:
    names: list[str] = []
    for block in _PYPROJECT.findall(text):
        names.extend(_QUOTED.findall(block))
    return names


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.lower()
        if key.isdigit() or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered
