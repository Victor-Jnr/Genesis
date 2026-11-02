"""Search public software sources, fetch documentation, and record what was found."""

from __future__ import annotations

import hashlib
import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from genesis.core.configuration.settings import Settings
from genesis.storage.database import Database
from genesis.tools.research.actions import ResearchAction
from genesis.tools.research.documents import html_to_text, read_documentation
from genesis.tools.research.http_client import HttpClient, HttpResponse, ResearchError, UrllibClient

_PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DDG_LINK = re.compile(
    r"<a[^>]*href=\"([^\"]+)\"[^>]*class='result-link'[^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET = re.compile(r"class='result-snippet'>(.*?)</td>", re.IGNORECASE | re.DOTALL)


class ResearchService:
    def __init__(self, settings: Settings, db: Database, root: Path, http: HttpClient | None = None) -> None:
        self.settings = settings
        self.db = db
        self.root = root
        research = settings.research
        self.http = http or UrllibClient(
            timeout=research.timeout_seconds,
            max_bytes=research.max_bytes,
            user_agent=research.user_agent,
            allow_private_hosts=research.allow_private_hosts,
        )
        self.download_directory = settings.resolve_path(research.download_directory, root)

    def execute(self, objective_id: str, action: ResearchAction) -> str:
        if not self.settings.research.enabled:
            return "Research is disabled in the configuration."
        try:
            if action.name == "search":
                return self._search(objective_id, _required(action.query, "query"))
            if action.name == "repositories":
                return self._repositories(objective_id, _required(action.query, "query"))
            if action.name == "package":
                return self._package(objective_id, _package_name(action.package))
            if action.name == "fetch":
                return self._fetch(objective_id, action.url)
            if action.name == "read":
                return self._read(objective_id, action.url)
            if action.name == "download":
                return self._download(objective_id, action.url)
        except ResearchError as exc:
            return f"Research failed: {exc}"
        return f"Research failed: unknown action '{action.name}'."

    def _search(self, objective_id: str, query: str) -> str:
        url = "https://lite.duckduckgo.com/lite/?" + urlencode({"q": query})
        response = self._get(url)
        if response.status >= 400:
            raise ResearchError(f"HTTP {response.status} for {response.url}")
        hits = _duckduckgo_hits(response.body.decode("utf-8", "replace"), self.settings.research.max_results)
        self._store(objective_id, "search", query, url, {"query": query, "hits": hits})
        if not hits:
            return f"No web results for {query}."
        lines = [f"Web results for {query}:"]
        for hit in hits:
            lines.append(f"- {hit['title']} — {hit['url']} — {hit['snippet']}")
        return "\n".join(lines)

    def _repositories(self, objective_id: str, query: str) -> str:
        repos: list[dict[str, str]] = []
        problems: list[str] = []
        github_url = "https://api.github.com/search/repositories?" + urlencode(
            {"q": query, "per_page": str(self.settings.research.max_results), "sort": "stars"}
        )
        gitlab_url = "https://gitlab.com/api/v4/projects?" + urlencode(
            {"search": query, "simple": "true", "per_page": str(self.settings.research.max_results)}
        )
        try:
            repos.extend(self._github_repositories(github_url))
        except ResearchError as exc:
            problems.append(str(exc))
        try:
            repos.extend(self._gitlab_repositories(gitlab_url))
        except ResearchError as exc:
            problems.append(str(exc))
        deduped = _dedupe_repos(repos)[: self.settings.research.max_results]
        self._store(
            objective_id,
            "repository",
            query,
            github_url,
            {"query": query, "repositories": deduped, "problems": problems},
        )
        if not deduped and problems:
            raise ResearchError(" ".join(problems))
        if not deduped:
            return f"No repositories found for {query}."
        lines = [f"Repositories for {query}:"]
        for repo in deduped:
            lines.append(f"- {repo['name']} — {repo['url']} — {repo['description']}")
        if problems:
            lines.append("Notes: " + " ".join(problems))
        return "\n".join(lines)

    def _github_repositories(self, url: str) -> list[dict[str, str]]:
        headers = {"Accept": "application/vnd.github+json"}
        token = (self.settings.research.github_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = _json_object(self._get(url, headers=headers))
        repos: list[dict[str, str]] = []
        for item in (payload.get("items") or [])[: self.settings.research.max_results]:
            if not isinstance(item, dict):
                continue
            repos.append(
                {
                    "name": str(item.get("full_name") or ""),
                    "url": str(item.get("html_url") or ""),
                    "description": str(item.get("description") or ""),
                }
            )
        return repos

    def _gitlab_repositories(self, url: str) -> list[dict[str, str]]:
        payload = _json_value(self._get(url))
        if not isinstance(payload, list):
            raise ResearchError(f"{url} did not return a repository list.")
        repos: list[dict[str, str]] = []
        for item in payload[: self.settings.research.max_results]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path_with_namespace") or "")
            web = str(item.get("web_url") or "")
            if not web and path:
                web = f"https://gitlab.com/{path}"
            repos.append(
                {
                    "name": str(item.get("name_with_namespace") or path),
                    "url": web,
                    "description": str(item.get("description") or ""),
                }
            )
        return repos

    def _package(self, objective_id: str, name: str) -> str:
        url = f"https://pypi.org/pypi/{quote(name)}/json"
        payload = _json_object(self._get(url))
        info = payload.get("info") if isinstance(payload, dict) else None
        if not isinstance(info, dict):
            raise ResearchError(f"PyPI returned no package info for {name}.")
        requires = [str(item) for item in (info.get("requires_dist") or []) if item]
        dependencies = [item.split(";", 1)[0].strip() for item in requires]
        record = {
            "name": str(info.get("name") or name),
            "version": str(info.get("version") or ""),
            "summary": str(info.get("summary") or ""),
            "home_page": str(info.get("home_page") or info.get("project_url") or ""),
            "dependencies": dependencies,
            "requires_dist": requires,
        }
        self._store(objective_id, "package", record["name"], url, record)
        dependency_text = ", ".join(dependencies) if dependencies else "none listed"
        return (
            f"Package {record['name']} {record['version']}\n"
            f"Summary: {record['summary']}\n"
            f"Project: {record['home_page']}\n"
            f"Dependencies: {dependency_text}"
        )

    def _fetch(self, objective_id: str, url: str) -> str:
        response = self._get(url)
        text = _response_text(response)
        self._store(
            objective_id,
            "page",
            _title_from(text, url),
            response.url,
            {"content_type": response.headers.get("content-type", ""), "excerpt": text[:1500], "truncated": response.truncated},
        )
        return f"Fetched {response.url}\n\n{text[:2000]}"

    def _read(self, objective_id: str, url: str) -> str:
        response = self._get(url)
        text = _response_text(response)
        notes = read_documentation(text)
        self._store(
            objective_id,
            "documentation",
            _title_from(text, url),
            response.url,
            {
                "dependencies": list(notes.dependencies),
                "install_requirements": list(notes.install_requirements),
                "excerpt": notes.excerpt,
                "truncated": response.truncated,
            },
        )
        dependencies = ", ".join(notes.dependencies) if notes.dependencies else "none found"
        installs = "\n".join(notes.install_requirements) if notes.install_requirements else "none found"
        return (
            f"Documentation from {response.url}\n"
            f"Dependencies: {dependencies}\n"
            f"Install requirements:\n{installs}\n\n"
            f"{notes.excerpt}"
        )

    def _download(self, objective_id: str, url: str) -> str:
        response = self._get(url)
        digest = hashlib.sha256(response.body).hexdigest()
        filename = _safe_filename(response.url)
        directory = self.download_directory / digest[:16]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_bytes(response.body)
        self._store(
            objective_id,
            "download",
            filename,
            response.url,
            {"bytes": len(response.body), "sha256": digest, "truncated": response.truncated},
            path=str(path),
            sha256=digest,
        )
        return f"Downloaded {filename} ({len(response.body)} bytes) from {response.url} to {path}. sha256 {digest}."

    def _get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        target = url.strip()
        parts = urlsplit(target)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ResearchError("Only http and https URLs can be retrieved.")
        return self.http.get(target, headers=headers)

    def _store(
        self,
        objective_id: str,
        kind: str,
        title: str,
        url: str | None,
        metadata: dict[str, object],
        *,
        path: str | None = None,
        sha256: str | None = None,
    ) -> None:
        self.db.add_resource(
            objective_id,
            kind,
            title or kind,
            url,
            path,
            sha256,
            json.dumps(metadata),
        )


def _required(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        raise ResearchError(f"A {label} is required.")
    if len(text) > 300:
        raise ResearchError(f"The {label} is too long.")
    return text


def _package_name(value: str) -> str:
    name = _required(value, "package name")
    if _PACKAGE.fullmatch(name) is None:
        raise ResearchError(f"{name} is not a package name.")
    return name


def _json_object(response: HttpResponse) -> dict[str, object]:
    payload = _json_value(response)
    if not isinstance(payload, dict):
        raise ResearchError(f"{response.url} did not return a JSON object.")
    return payload


def _json_value(response: HttpResponse) -> object:
    if response.status >= 400:
        detail = response.body.decode("utf-8", "replace").strip()
        message = f"HTTP {response.status} for {response.url}"
        if detail:
            message = f"{message}: {detail[:300]}"
        raise ResearchError(message)
    try:
        return json.loads(response.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchError(f"{response.url} did not return JSON.") from exc


def _response_text(response: HttpResponse) -> str:
    if response.status >= 400:
        raise ResearchError(f"HTTP {response.status} for {response.url}")
    raw = response.body.decode("utf-8", "replace")
    content_type = response.headers.get("content-type", "")
    if "html" in content_type.lower() or raw.lstrip().startswith("<"):
        return html_to_text(raw)
    return raw.strip()


def _duckduckgo_hits(html: str, limit: int) -> list[dict[str, str]]:
    snippets = [html_to_text(unescape(item)) for item in _DDG_SNIPPET.findall(html)]
    hits: list[dict[str, str]] = []
    for index, match in enumerate(_DDG_LINK.finditer(html)):
        if len(hits) >= limit:
            break
        target = _duckduckgo_target(unescape(match.group(1)))
        if not target.startswith(("http://", "https://")):
            continue
        title = html_to_text(unescape(match.group(2))) or target
        snippet = snippets[index] if index < len(snippets) else ""
        hits.append({"title": title, "url": target, "snippet": snippet})
    return hits


def _duckduckgo_target(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    uddg = parse_qs(urlsplit(href).query).get("uddg", [""])[0]
    return unquote(uddg) if uddg else href


def _dedupe_repos(repos: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    ordered: list[dict[str, str]] = []
    for repo in repos:
        key = repo.get("url") or repo.get("name") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(repo)
    return ordered


def _title_from(text: str, url: str) -> str:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return (first or urlsplit(url).path.rsplit("/", 1)[-1] or url)[:160]


def _safe_filename(url: str) -> str:
    name = urlsplit(url).path.rsplit("/", 1)[-1] or "download"
    cleaned = _FILENAME.sub("_", name).strip("._")[:120]
    return cleaned or "download"
