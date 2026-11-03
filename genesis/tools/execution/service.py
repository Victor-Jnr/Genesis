"""Run one local program. A shell and public network targets stay unavailable."""

from __future__ import annotations

import ipaddress
import os
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from genesis.core.configuration.settings import Settings
from genesis.tools.research.actions import ResearchAction

_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_URL = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
_SHELL = ("|", ";", "&", "`", "$(", ">", "<", "\n", "\r")


class ExecutionError(ValueError):
    pass


class ExecutionService:
    def __init__(self, settings: Settings, root: Path, runner=None) -> None:
        self.settings = settings
        self.root = root.resolve()
        self.runner = runner or run_program

    def execute(self, _objective_id: str | None, action: ResearchAction) -> str:
        try:
            return self.run(action.command or action.task)
        except ExecutionError as exc:
            return f"Execution failed: {exc}"

    def run(self, command: str) -> str:
        if not self.settings.execution.enabled:
            raise ExecutionError("Local commands are disabled. Set execution.enabled in the configuration.")
        argv = split_command(command)
        check_targets(argv, allow_private=self.settings.lab.enabled)
        timeout = self.settings.execution.timeout_seconds
        try:
            code, output = self.runner(argv, self.root, timeout)
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(f"The command exceeded {timeout:g} seconds.") from exc
        except OSError as exc:
            raise ExecutionError(str(exc)) from exc
        limit = self.settings.execution.max_output_chars
        visible = output.strip()
        if len(visible) > limit:
            visible = visible[:limit] + "\n... truncated"
        return f"Exit {code}\n{visible or '(no output)'}"


def split_command(command: str) -> list[str]:
    text = command.strip()
    if not text:
        raise ExecutionError("A command is required.")
    if any(token in text for token in _SHELL):
        raise ExecutionError("The command must be one program and its arguments.")
    argv = _split_windows(text) if os.name == "nt" else shlex.split(text, posix=True)
    if not argv:
        raise ExecutionError("A command is required.")
    if len(argv) > 24:
        raise ExecutionError("The command has too many arguments.")
    return argv


def _split_windows(text: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    quoted = False
    for char in text:
        if char == '"':
            quoted = not quoted
            continue
        if char.isspace() and not quoted:
            if buffer:
                parts.append("".join(buffer))
                buffer = []
            continue
        buffer.append(char)
    if quoted:
        raise ExecutionError("The command has an unfinished quote.")
    if buffer:
        parts.append("".join(buffer))
    return parts


def check_targets(argv: list[str], *, allow_private: bool) -> None:
    text = " ".join(argv)
    for match in _IP.finditer(text):
        _allow_address(ipaddress.ip_address(match.group(0)), allow_private=allow_private)
    for match in _URL.finditer(text):
        host = urlsplit(match.group(0)).hostname
        if not host:
            raise ExecutionError("The command contains a URL without a host.")
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ExecutionError("The command can name an address, not a public site. Use fetch or browse.") from exc
        _allow_address(address, allow_private=allow_private)


def _allow_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private: bool) -> None:
    if address.is_loopback:
        return
    if address.is_private or address.is_link_local or address.is_reserved:
        if allow_private:
            return
        raise ExecutionError("A private address needs lab.enabled in the configuration.")
    raise ExecutionError("That address is outside the lab network.")


def run_program(argv: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return completed.returncode, output
