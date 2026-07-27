"""GitHub connector.

Bound to a single repository at construction. That is a deliberate scoping choice: a
run targets one repo, so the subagent never gets to name a repo, and a confused or
injected agent cannot write to a different one. Least privilege (FR-T2) is easier to
enforce by removing the parameter than by validating it.

Credentials come from `GITHUB_TOKEN` if set, otherwise from the `gh` CLI keyring. The
token is never passed into a prompt.
"""

from __future__ import annotations

import base64
import subprocess
from functools import cached_property
from typing import Any

import httpx

from conductor.connectors.base import Connector, ConnectorError, ToolSpec

API_ROOT = "https://api.github.com"


def _token_from_gh_cli() -> str | None:
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    token = out.stdout.strip()
    return token or None


def resolve_token(explicit: str | None = None) -> str | None:
    import os

    return explicit or os.environ.get("GITHUB_TOKEN") or _token_from_gh_cli()


class GitHubConnector(Connector):
    name = "github"

    def __init__(self, repo: str, token: str | None = None, *, timeout: float = 30.0) -> None:
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self._token = resolve_token(token)
        self._timeout = timeout

    @cached_property
    def _client(self) -> httpx.Client:
        if not self._token:
            raise ConnectorError(
                "no GitHub credentials — set GITHUB_TOKEN or run `gh auth login`"
            )
        return httpx.Client(
            base_url=API_ROOT,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    # ---- Connector interface ---------------------------------------------

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="github.get_issue",
                description="Fetch an issue's title, body, labels, and state",
                parameters={"number": "issue number (int)"},
            ),
            ToolSpec(
                name="github.list_files",
                description="List files in a repository directory at a ref",
                parameters={"path": "directory path, '' for root", "ref": "branch or sha"},
                optional=["path", "ref"],
            ),
            ToolSpec(
                name="github.read_file",
                description="Read a file's contents at a ref",
                parameters={"path": "file path", "ref": "branch or sha"},
                optional=["ref"],
            ),
            ToolSpec(
                name="github.get_pr_diff",
                description="Fetch the unified diff for a pull request",
                parameters={"number": "PR number (int)"},
            ),
            ToolSpec(
                name="github.create_branch",
                description="Create a branch from a base ref",
                parameters={"name": "new branch name", "from_ref": "base branch"},
                optional=["from_ref"],
                mutates=True,
                reversible=True,
            ),
            ToolSpec(
                name="github.write_file",
                description="Create or update a file on a branch, committing the change",
                parameters={
                    "path": "file path",
                    "content": "full new file contents",
                    "branch": "target branch",
                    "message": "commit message",
                },
                mutates=True,
                reversible=True,
            ),
            ToolSpec(
                name="github.open_pr",
                description="Open a pull request for human review. Does not merge.",
                parameters={
                    "head": "source branch",
                    "base": "target branch",
                    "title": "PR title",
                    "body": "PR description",
                },
                optional=["body"],
                mutates=True,
                reversible=True,
            ),
        ]

    def health(self) -> tuple[bool, str]:
        if not self._token:
            return False, "no GitHub credentials (set GITHUB_TOKEN or run `gh auth login`)"
        try:
            r = self._client.get(f"/repos/{self.repo}")
        except httpx.HTTPError as exc:
            return False, f"network error: {exc}"
        if r.status_code == 200:
            return True, f"authenticated, {self.repo} reachable"
        return False, f"{self.repo} -> HTTP {r.status_code}"

    def invoke(self, tool: str, arguments: dict[str, Any]) -> Any:
        handler = {
            "github.get_issue": self._get_issue,
            "github.list_files": self._list_files,
            "github.read_file": self._read_file,
            "github.get_pr_diff": self._get_pr_diff,
            "github.create_branch": self._create_branch,
            "github.write_file": self._write_file,
            "github.open_pr": self._open_pr,
        }.get(tool)
        if handler is None:
            raise KeyError(f"github connector has no tool {tool!r}")
        try:
            return handler(**arguments)
        except TypeError as exc:
            raise ConnectorError(f"{tool}: bad arguments: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"{tool}: network error: {exc}") from exc

    # ---- helpers ----------------------------------------------------------

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        r = self._client.request(method, path, **kw)
        if r.status_code >= 400:
            raise ConnectorError(f"GitHub {method} {path} -> {r.status_code}: {r.text[:400]}")
        return r

    # ---- read tools -------------------------------------------------------

    def _get_issue(self, number: int) -> dict[str, Any]:
        d = self._request("GET", f"/repos/{self.repo}/issues/{int(number)}").json()
        return {
            "number": d["number"],
            "title": d["title"],
            "body": d.get("body") or "",
            "state": d["state"],
            "labels": [x["name"] for x in d.get("labels", [])],
            "url": d["html_url"],
        }

    def _list_files(self, path: str = "", ref: str | None = None) -> list[dict[str, Any]]:
        params = {"ref": ref} if ref else {}
        d = self._request(
            "GET", f"/repos/{self.repo}/contents/{path.lstrip('/')}", params=params
        ).json()
        if isinstance(d, dict):  # a file path, not a directory
            d = [d]
        return [{"path": e["path"], "type": e["type"], "size": e.get("size")} for e in d]

    def _read_file(self, path: str, ref: str | None = None) -> dict[str, Any]:
        params = {"ref": ref} if ref else {}
        d = self._request(
            "GET", f"/repos/{self.repo}/contents/{path.lstrip('/')}", params=params
        ).json()
        if d.get("type") != "file":
            raise ConnectorError(f"{path} is not a file")
        content = base64.b64decode(d["content"]).decode("utf-8", errors="replace")
        return {"path": d["path"], "sha": d["sha"], "content": content}

    def _get_pr_diff(self, number: int) -> str:
        r = self._client.get(
            f"/repos/{self.repo}/pulls/{int(number)}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        if r.status_code >= 400:
            raise ConnectorError(f"get_pr_diff -> {r.status_code}: {r.text[:400]}")
        return r.text

    # ---- write tools ------------------------------------------------------

    def _create_branch(self, name: str, from_ref: str = "main") -> dict[str, Any]:
        base = self._request("GET", f"/repos/{self.repo}/git/ref/heads/{from_ref}").json()
        d = self._request(
            "POST",
            f"/repos/{self.repo}/git/refs",
            json={"ref": f"refs/heads/{name}", "sha": base["object"]["sha"]},
        ).json()
        return {"branch": name, "sha": d["object"]["sha"], "from": from_ref}

    def _write_file(
        self, path: str, content: str, branch: str, message: str
    ) -> dict[str, Any]:
        path = path.lstrip("/")
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch,
        }
        # Updating requires the blob sha; creating must omit it.
        existing = self._client.get(
            f"/repos/{self.repo}/contents/{path}", params={"ref": branch}
        )
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]

        d = self._request("PUT", f"/repos/{self.repo}/contents/{path}", json=payload).json()
        return {
            "path": path,
            "branch": branch,
            "commit": d["commit"]["sha"],
            "created": existing.status_code != 200,
        }

    def _open_pr(self, head: str, base: str, title: str, body: str = "") -> dict[str, Any]:
        d = self._request(
            "POST",
            f"/repos/{self.repo}/pulls",
            json={"head": head, "base": base, "title": title, "body": body},
        ).json()
        return {"number": d["number"], "url": d["html_url"], "state": d["state"]}
