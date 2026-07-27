import pytest

from conductor.audit import AuditStore, EventType
from conductor.connectors import Connector, ConnectorRegistry, ScopedToolbox, ToolSpec
from conductor.connectors.github import GitHubConnector
from conductor.subagents import SubagentRegistry


class FakeConnector(Connector):
    name = "fake"

    def __init__(self):
        self.calls = []

    def tools(self):
        return [
            ToolSpec(name="fake.read", description="read something"),
            ToolSpec(name="fake.write", description="write something", mutates=True),
            ToolSpec(
                name="fake.destroy",
                description="delete something",
                mutates=True,
                reversible=False,
            ),
        ]

    def invoke(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"ok": tool}


@pytest.fixture
def registry():
    r = ConnectorRegistry()
    r.register(FakeConnector())
    return r


@pytest.fixture
def audit(tmp_path):
    s = AuditStore(tmp_path / "t.db")
    yield s
    s.close()


def test_risk_classification_feeds_the_policy_engine(registry):
    assert registry.spec("fake.read").qualified_risk == "read"
    assert registry.spec("fake.write").qualified_risk == "reversible_write"
    assert registry.spec("fake.destroy").qualified_risk == "irreversible"


def test_duplicate_connector_registration_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeConnector())


def test_scoped_toolbox_allows_declared_tools(registry, audit):
    from conductor.subagents import SubagentManifest

    manifest = SubagentManifest(
        name="agent", role="r", route="tester", prompt="p", tools=["fake.read"]
    )
    run_id = audit.start_run("goal")
    box = ScopedToolbox(registry, manifest, audit, run_id)

    assert box.invoke("fake.read", {}) == {"ok": "fake.read"}


def test_scoped_toolbox_blocks_undeclared_tools(registry, audit):
    """This is the FR-T2 enforcement point. A prompt cannot talk past it."""
    from conductor.subagents import SubagentManifest

    manifest = SubagentManifest(
        name="agent", role="r", route="tester", prompt="p", tools=["fake.read"]
    )
    run_id = audit.start_run("goal")
    box = ScopedToolbox(registry, manifest, audit, run_id)

    with pytest.raises(PermissionError, match="not permitted"):
        box.invoke("fake.write", {})


def test_blocked_call_is_not_executed(registry, audit):
    from conductor.subagents import SubagentManifest

    connector = registry.connectors()[0]
    manifest = SubagentManifest(
        name="agent", role="r", route="tester", prompt="p", tools=["fake.read"]
    )
    run_id = audit.start_run("goal")
    box = ScopedToolbox(registry, manifest, audit, run_id)

    with pytest.raises(PermissionError):
        box.invoke("fake.destroy", {})
    assert connector.calls == []


def test_permitted_calls_are_audited(registry, audit):
    from conductor.subagents import SubagentManifest

    manifest = SubagentManifest(
        name="agent", role="r", route="tester", prompt="p", tools=["fake.read"]
    )
    run_id = audit.start_run("goal")
    ScopedToolbox(registry, manifest, audit, run_id).invoke("fake.read", {"x": 1})

    types = [e.event_type for e in audit.events(run_id)]
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types


def test_shipped_manifests_only_reference_real_tools():
    """Boot-time guard: a typo in a manifest must fail here, not mid-run."""
    conn_registry = ConnectorRegistry()
    conn_registry.register(GitHubConnector("owner/repo", token="fake"))

    subagents = SubagentRegistry.from_dir("subagents")
    for manifest in subagents.all():
        missing = conn_registry.validate_manifest(manifest)
        assert not missing, f"{manifest.name} declares unknown tools: {missing}"


def test_github_connector_requires_owner_slash_name():
    with pytest.raises(ValueError, match="owner/name"):
        GitHubConnector("justarepo")


def test_github_reviewer_scope_has_no_mutating_tools():
    """Cross-check the manifest against the connector's own risk flags."""
    conn_registry = ConnectorRegistry()
    conn_registry.register(GitHubConnector("owner/repo", token="fake"))
    reviewer = SubagentRegistry.from_dir("subagents").get("reviewer")

    for tool in reviewer.tools:
        assert not conn_registry.spec(tool).mutates, f"reviewer has mutating tool {tool}"


def test_github_implementer_has_no_irreversible_tools():
    """Implementer may write, but merge/delete must never be in reach (FR-H4)."""
    conn_registry = ConnectorRegistry()
    conn_registry.register(GitHubConnector("owner/repo", token="fake"))
    implementer = SubagentRegistry.from_dir("subagents").get("implementer")

    for tool in implementer.tools:
        assert conn_registry.spec(tool).reversible, f"implementer has irreversible tool {tool}"
