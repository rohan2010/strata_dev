import pytest

from conductor.models import ModelRegistry, Provider
from conductor.secrets import (
    PROVIDER_ENV_VARS,
    credential_for,
    has_credential,
    load_env,
    missing_providers,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for variables in PROVIDER_ENV_VARS.values():
        for var in variables:
            monkeypatch.delenv(var, raising=False)


def test_every_provider_has_an_entry():
    """A provider with no entry would silently report as credentialed."""
    assert set(PROVIDER_ENV_VARS) == set(Provider)


def test_credential_reports_which_variable_supplied_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert credential_for(Provider.ANTHROPIC) == "ANTHROPIC_API_KEY"
    assert has_credential(Provider.ANTHROPIC)


def test_gemini_accepts_either_variable_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert credential_for(Provider.GOOGLE) == "GEMINI_API_KEY"


def test_google_api_key_wins_when_both_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "b")
    assert credential_for(Provider.GOOGLE) == "GOOGLE_API_KEY"


def test_ollama_needs_no_credential():
    assert has_credential(Provider.OLLAMA)
    assert credential_for(Provider.OLLAMA) is None


def test_missing_providers_lists_only_uncredentialed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    missing = missing_providers({Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA})
    assert missing == [Provider.OPENAI]


def test_exported_variable_beats_the_dotfile(tmp_path, monkeypatch):
    """A stale .env silently shadowing an exported key is the bug this prevents."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=from-dotfile\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    assert load_env(env) is True
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_dotfile_fills_an_unset_variable(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=from-dotfile\n")
    assert load_env(env) is True

    import os

    assert os.environ["OPENAI_API_KEY"] == "from-dotfile"


def test_missing_dotfile_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") is False


def test_shipped_routes_only_reach_providers_we_can_name_variables_for():
    """Boot guard: a new provider in models.yaml must come with its env var mapping."""
    registry = ModelRegistry.from_file("models.yaml")
    reachable = {
        registry.models[a].provider for chain in registry.routes.values() for a in chain
    }
    assert reachable <= set(PROVIDER_ENV_VARS)
