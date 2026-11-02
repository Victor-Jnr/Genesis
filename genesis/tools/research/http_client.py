"""Fetch public HTTP resources with a size limit and no private-host redirects."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes
    truncated: bool


class HttpClient(Protocol):
    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class UrllibClient:
    def __init__(self, *, timeout: float, max_bytes: int, user_agent: str, allow_private_hosts: bool) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.allow_private_hosts = allow_private_hosts
        self._opener = build_opener(_NoRedirect)

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        current = url
        for _ in range(4):
            validate_public_url(current, allow_private_hosts=self.allow_private_hosts)
            request = Request(
                current,
                headers={"User-Agent": self.user_agent, "Accept": "*/*", **(headers or {})},
            )
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = response.read(self.max_bytes + 1)
                    header_map = {key.lower(): value for key, value in response.headers.items()}
                    return HttpResponse(
                        url=response.geturl(),
                        status=getattr(response, "status", 200),
                        headers=header_map,
                        body=payload[: self.max_bytes],
                        truncated=len(payload) > self.max_bytes,
                    )
            except HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    location = exc.headers.get("Location")
                    if not location:
                        raise ResearchError(f"HTTP {exc.code} from {current} had no redirect location.") from exc
                    current = urljoin(current, location)
                    continue
                detail = exc.read(300).decode("utf-8", "replace").strip()
                message = f"HTTP {exc.code} for {current}"
                if detail:
                    message = f"{message}: {detail}"
                raise ResearchError(message) from exc
            except URLError as exc:
                raise ResearchError(f"Could not retrieve {current}: {exc.reason}") from exc
        raise ResearchError(f"Too many redirects from {url}.")


def validate_public_url(url: str, *, allow_private_hosts: bool) -> None:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ResearchError("Only http and https URLs can be retrieved.")
    if allow_private_hosts:
        return
    host = parts.hostname
    literal = _literal_ip(host)
    addresses = [literal] if literal is not None else _resolved_addresses(host)
    if any(_blocked(address) for address in addresses):
        raise ResearchError(f"{host} is not a public host.")


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ResearchError(f"Could not resolve {host}.") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        addresses.append(ipaddress.ip_address(info[4][0]))
    if not addresses:
        raise ResearchError(f"Could not resolve {host}.")
    return addresses


def _blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )
