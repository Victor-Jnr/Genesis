"""HTML workstation. The pages read the JSON API in this same process."""

from __future__ import annotations

from pathlib import Path
from string import Template

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_LAYOUT = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Genesis — $title</title>
  <link rel="stylesheet" href="/workstation/assets/workstation.css?v=3">
</head>
<body data-page="$page">
  <div class="shell">
    <nav class="nav" aria-label="Workstation">
      <p>Genesis</p>
      $nav
      <a href="/docs">API docs</a>
    </nav>
    <main class="main">
      <h1>$title</h1>
      <p class="lede">$lede</p>
      <div data-error></div>
      $body
    </main>
  </div>
  <script src="/workstation/assets/workstation.js?v=3"></script>
</body>
</html>
"""
)

_NAV = (
    ("dashboard", "/workstation", "Dashboard"),
    ("monitor", "/workstation/monitor", "Monitor"),
    ("objectives", "/workstation/objectives", "Objectives"),
    ("development", "/workstation/development", "Development"),
    ("capabilities", "/workstation/capabilities", "Capabilities"),
)


def register_workstation(app: FastAPI) -> None:
    static = Path(__file__).resolve().parent / "static"
    app.mount("/workstation/assets", StaticFiles(directory=static), name="workstation-assets")

    @app.get("/workstation", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return _page(
            "dashboard",
            "Dashboard",
            "The model, what is running, and the latest activity.",
            """
            <section id="status" class="grid" aria-label="Status"></section>
            <section class="block">
              <form id="chat-form">
                <h2>Chat</h2>
                <label for="chat-message">Message</label>
                <textarea id="chat-message" name="message" required></textarea>
                <button type="submit">Send</button>
              </form>
              <div id="chat-log" aria-live="polite"></div>
            </section>
            <section class="block">
              <form id="objective-form">
                <h2>New objective</h2>
                <label for="objective-request">Request</label>
                <textarea id="objective-request" name="request" required></textarea>
                <button type="submit">Start</button>
              </form>
            </section>
            <section class="block"><h2>Objectives</h2><div id="objectives"></div></section>
            <section class="block"><h2>Generation</h2><div id="generation" class="grid"></div></section>
            <section class="block"><h2>Recent activity</h2><div id="activity"></div></section>
            """,
        )

    @app.get("/workstation/monitor", response_class=HTMLResponse)
    def monitor() -> HTMLResponse:
        return _page(
            "monitor",
            "Monitor",
            "Events from an objective, refreshed while this page is open.",
            """
            <section class="block" id="monitor-list"></section>
            <section class="block" id="monitor-detail"></section>
            """,
        )

    @app.get("/workstation/objectives", response_class=HTMLResponse)
    def objectives() -> HTMLResponse:
        return _page(
            "objectives",
            "Objectives",
            "Ongoing and finished objectives.",
            '<section class="block" id="objective-list"></section>',
        )

    @app.get("/workstation/objectives/{objective_id}", response_class=HTMLResponse)
    def objective(objective_id: str) -> HTMLResponse:
        return _page(
            "objective",
            "Objective",
            "Steps, events, experiments, and sources for this run.",
            f'<section class="block" id="objective" data-id="{_attr(objective_id)}"></section>',
        )

    @app.get("/workstation/development", response_class=HTMLResponse)
    def development() -> HTMLResponse:
        return _page(
            "development",
            "Development",
            "Project files and the archive of tested generations.",
            """
            <section class="block">
              <form id="source-form">
                <label for="path">Path</label>
                <input id="path" name="path" type="text" value="genesis" required>
                <button type="submit">Open</button>
              </form>
            </section>
            <section class="block" id="source"></section>
            <section class="block"><h2>Archive</h2><div id="archive"></div></section>
            """,
        )

    @app.get("/workstation/capabilities", response_class=HTMLResponse)
    def capabilities() -> HTMLResponse:
        return _page(
            "capabilities",
            "Capabilities",
            "Built-in actions, installed modules, saved skills, and the running configuration.",
            """
            <section class="block"><h2>Actions and modules</h2><div id="capabilities"></div></section>
            <section class="block"><h2>Installed modules</h2><div id="modules"></div></section>
            <section class="block"><h2>Skills</h2><div id="skills"></div></section>
            <section class="block">
              <h2>Configuration</h2>
              <p class="muted">Read from the process that is running. Edit config/genesis.yaml and restart to change these values.</p>
              <div id="config"></div>
            </section>
            <section class="block"><h2>Model files</h2><div id="models"></div></section>
            """,
        )


def _attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _page(page: str, title: str, lede: str, body: str) -> HTMLResponse:
    current = "objectives" if page == "objective" else page
    links = []
    for name, href, label in _NAV:
        marker = ' aria-current="page"' if name == current else ""
        links.append(f'<a href="{href}"{marker}>{label}</a>')
    html = _LAYOUT.substitute(title=title, page=page, nav="\n".join(links), lede=lede, body=body)
    return HTMLResponse(html)
