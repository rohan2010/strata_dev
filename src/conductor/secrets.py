"""Credential loading, and the one place that knows which env var belongs to whom.

Keys are never passed into a prompt and never stored in `models.yaml` — that file is
committed, and a routing table is exactly the kind of thing people paste into issues. The
provider packages read their own env vars at model construction; this module's job is to
make sure `.env` is loaded before that happens, and to report what is missing *before* a
run burns a planning call discovering it.

EXECUTION-PLAN.md section 6 wants secrets in a vault rather than a dotfile from the first
connector onward. `.env` is the Phase 0 stand-in; `credential_status` is the seam a vault
backend slots into, since nothing else asks where a key came from.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from conductor.models.providers import Provider

#: Env vars each provider reads, in the order the provider itself checks them.
#: Verified against the installed integration packages, not assumed.
PROVIDER_ENV_VARS: dict[Provider, tuple[str, ...]] = {
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
    Provider.OPENAI: ("OPENAI_API_KEY",),
    Provider.GOOGLE: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    Provider.OLLAMA: (),  # local; reachability, not credentials
}

#: Read by the GitHub connector, which falls back to the `gh` CLI keyring.
GITHUB_ENV_VAR = "GITHUB_TOKEN"


def load_env(path: Path | str = ".env") -> bool:
    """Load `.env` into the environment if present. Returns whether a file was found.

    `override=False` so an explicitly exported variable always beats the dotfile — the
    surprising alternative is a stale `.env` silently shadowing the key you just exported.
    """
    path = Path(path)
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    return True


def credential_for(provider: Provider) -> str | None:
    """The env var actually supplying this provider's key, or None if none is set."""
    for var in PROVIDER_ENV_VARS.get(provider, ()):
        if os.environ.get(var):
            return var
    return None


def has_credential(provider: Provider) -> bool:
    return provider is Provider.OLLAMA or credential_for(provider) is not None


def missing_providers(providers: set[Provider]) -> list[Provider]:
    return sorted((p for p in providers if not has_credential(p)), key=str)
