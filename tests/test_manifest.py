import pytest
from pydantic import ValidationError

from conductor.subagents import RiskProfile, SubagentManifest, SubagentRegistry

MINIMAL = {
    "name": "tester",
    "role": "Author and run tests",
    "route": "tester",
    "prompt": "You write tests.",
}


def test_manifest_defaults():
    m = SubagentManifest.model_validate(MINIMAL)
    assert m.risk_profile is RiskProfile.READ_ONLY
    assert m.effort == "high"
    assert m.tools == []


@pytest.mark.parametrize(
    "bad_route",
    ["claude-opus-5", "anthropic:claude-opus-5", "gpt-5.2", "gemini-3-pro"],
)
def test_model_id_in_route_field_is_rejected(bad_route):
    """A model id here would fall through to default_route and quietly run the wrong model."""
    with pytest.raises(ValidationError, match="looks like a model id"):
        SubagentManifest.model_validate(MINIMAL | {"route": bad_route})


def test_dated_model_ids_are_no_longer_the_manifest_s_business():
    """Regression guard for the multi-provider move.

    `gpt-4o-2024-08-06` is the *recommended* pinned form on OpenAI. The old
    date-suffix rejection would have made that unusable, so it is gone — and the
    manifest no longer names model ids at all.
    """
    assert "model" not in SubagentManifest.model_fields
    SubagentManifest.model_validate(MINIMAL | {"route": "implementer"})


def test_tool_allow_list_is_the_enforcement_point():
    m = SubagentManifest.model_validate(MINIMAL | {"tools": ["github.read_file"]})
    assert m.allows("github.read_file")
    assert not m.allows("github.write_file")


def test_registry_loads_shipped_manifests():
    reg = SubagentRegistry.from_dir("subagents")
    assert reg.names() == ["implementer", "reviewer"]
    assert reg.get("reviewer").risk_profile is RiskProfile.READ_ONLY
    assert reg.get("implementer").risk_profile is RiskProfile.WRITE


def test_reviewer_has_no_write_tools():
    """FR-T2 least privilege — asserted, not assumed."""
    reviewer = SubagentRegistry.from_dir("subagents").get("reviewer")
    forbidden = {"write_file", "commit", "open_pr", "create_branch", "merge"}
    assert not any(any(f in tool for f in forbidden) for tool in reviewer.tools)


def test_capability_summary_lists_every_subagent():
    """The planner routes off this string, so every registered subagent must appear."""
    reg = SubagentRegistry.from_dir("subagents")
    summary = reg.capability_summary()
    for name in reg.names():
        assert name in summary


def test_duplicate_names_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("name: dup\nrole: r\nroute: tester\nprompt: p\n")
    (tmp_path / "b.yaml").write_text("name: dup\nrole: r2\nroute: tester\nprompt: p\n")
    with pytest.raises(ValueError, match="duplicate subagent name"):
        SubagentRegistry.from_dir(tmp_path)


def test_shipped_manifests_and_model_routes_agree():
    """models.yaml and the manifests must agree at boot, not mid-run."""
    from conductor.models import ModelRegistry

    models = ModelRegistry.from_file("models.yaml")
    subagents = SubagentRegistry.from_dir("subagents")

    assert subagents.validate_routes(models) == []
    for m in subagents.all():
        assert models.has_route(m.route), m.name


def test_unknown_subagent_error_lists_known_ones():
    reg = SubagentRegistry.from_dir("subagents")
    with pytest.raises(KeyError, match="implementer, reviewer"):
        reg.get("nonexistent")
