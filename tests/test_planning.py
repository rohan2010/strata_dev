import pytest
from pydantic import ValidationError

from conductor.audit import AuditStore, EventType
from conductor.models import ModelResult, ModelSpec
from conductor.planning import Planner, PlanningError, Task, TaskGraph
from conductor.subagents import SubagentRegistry


def graph(**kw) -> TaskGraph:
    base = {
        "goal": "fix login",
        "tasks": [
            {"id": "impl", "subagent": "implementer", "objective": "fix it"},
            {"id": "rev", "subagent": "reviewer", "objective": "review it", "depends_on": ["impl"]},
        ],
    }
    return TaskGraph.model_validate(base | kw)


# ---- task graph ----------------------------------------------------------


def test_layers_expose_parallelism():
    g = TaskGraph.model_validate(
        {
            "goal": "g",
            "tasks": [
                {"id": "a", "subagent": "implementer", "objective": "o"},
                {"id": "b", "subagent": "reviewer", "objective": "o"},
                {"id": "c", "subagent": "reviewer", "objective": "o", "depends_on": ["a", "b"]},
            ],
        }
    )
    assert [[t.id for t in layer] for layer in g.layers()] == [["a", "b"], ["c"]]
    assert g.width == 2


def test_serial_chain_has_width_one():
    assert graph().width == 1


def test_cycles_are_rejected_at_construction():
    with pytest.raises(ValidationError, match="dependency cycle"):
        TaskGraph.model_validate(
            {
                "goal": "g",
                "tasks": [
                    {"id": "a", "subagent": "x", "objective": "o", "depends_on": ["b"]},
                    {"id": "b", "subagent": "x", "objective": "o", "depends_on": ["a"]},
                ],
            }
        )


def test_dependency_on_unknown_task_is_rejected():
    with pytest.raises(ValidationError, match="unknown task"):
        TaskGraph.model_validate(
            {
                "goal": "g",
                "tasks": [{"id": "a", "subagent": "x", "objective": "o", "depends_on": ["ghost"]}],
            }
        )


def test_duplicate_task_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate task ids"):
        TaskGraph.model_validate(
            {
                "goal": "g",
                "tasks": [
                    {"id": "a", "subagent": "x", "objective": "o"},
                    {"id": "a", "subagent": "y", "objective": "o"},
                ],
            }
        )


def test_self_dependency_rejected():
    with pytest.raises(ValidationError, match="depends on itself"):
        TaskGraph.model_validate(
            {"goal": "g", "tasks": [{"id": "a", "subagent": "x", "objective": "o",
                                     "depends_on": ["a"]}]}
        )


def test_ready_respects_dependencies_and_in_flight():
    g = graph()
    assert [t.id for t in g.ready(completed=set())] == ["impl"]
    assert g.ready(completed=set(), started={"impl"}) == []
    assert [t.id for t in g.ready(completed={"impl"}, started={"impl"})] == ["rev"]


def test_blocked_by_is_transitive():
    g = TaskGraph.model_validate(
        {
            "goal": "g",
            "tasks": [
                {"id": "a", "subagent": "x", "objective": "o"},
                {"id": "b", "subagent": "x", "objective": "o", "depends_on": ["a"]},
                {"id": "c", "subagent": "x", "objective": "o", "depends_on": ["b"]},
                {"id": "d", "subagent": "x", "objective": "o"},
            ],
        }
    )
    assert g.blocked_by({"a"}) == {"b", "c"}


def test_unknown_subagents_reported_by_task():
    assert graph().unknown_subagents({"implementer"}) == {"rev": "reviewer"}


# ---- planner -------------------------------------------------------------


class FakePlannerClient:
    """Returns a queued sequence of parsed objects, one per invoke."""

    def __init__(self, *queued):
        self.queued = list(queued)
        self.calls = []

    def invoke(self, role, messages, **kw):
        self.calls.append((role, messages, kw))
        item = self.queued.pop(0)
        if isinstance(item, Exception):
            raise item
        spec = ModelSpec(alias="fake", provider="anthropic", id="x")
        return ModelResult(text="", spec=spec, parsed=item)


@pytest.fixture
def subagents():
    return SubagentRegistry.from_dir("subagents")


@pytest.fixture
def audit(tmp_path):
    s = AuditStore(tmp_path / "t.db")
    yield s
    s.close()


def test_plan_is_returned_and_audited(subagents, audit):
    run_id = audit.start_run("goal")
    client = FakePlannerClient(graph())
    plan = Planner(client, subagents, audit=audit, run_id=run_id).plan("fix login")

    assert [t.id for t in plan.tasks] == ["impl", "rev"]
    events = [e for e in audit.events(run_id) if e.event_type is EventType.PLAN_CREATED]
    assert len(events) == 1
    assert events[0].payload["task_count"] == 2
    assert events[0].payload["parallel_width"] == 1


def test_prompt_is_built_from_the_registry_not_hardcoded(subagents):
    """FR-S4/FR-O3: adding a subagent must change the planner's options with no code edit."""
    prompt = Planner(FakePlannerClient(), subagents).system_prompt()
    for name in subagents.names():
        assert name in prompt


def test_unregistered_subagent_triggers_one_repair_then_succeeds(subagents, audit):
    bad = TaskGraph.model_validate(
        {"goal": "g", "tasks": [{"id": "a", "subagent": "ghostwriter", "objective": "o"}]}
    )
    client = FakePlannerClient(bad, graph())
    plan = Planner(client, subagents, audit=audit, run_id=audit.start_run("g")).plan("fix login")

    assert len(client.calls) == 2
    assert [t.id for t in plan.tasks] == ["impl", "rev"]
    # The repair turn must name the real roster, or the model cannot correct itself.
    repair_messages = client.calls[1][1]
    assert "ghostwriter" in repair_messages[-1][1]
    assert "implementer, reviewer" in repair_messages[-1][1]


def test_planner_gives_up_after_one_repair(subagents, audit):
    bad = TaskGraph.model_validate(
        {"goal": "g", "tasks": [{"id": "a", "subagent": "ghostwriter", "objective": "o"}]}
    )
    client = FakePlannerClient(bad, bad)
    planner = Planner(client, subagents, audit=audit, run_id=audit.start_run("g"))

    with pytest.raises(PlanningError, match="after one repair"):
        planner.plan("fix login")
    assert len(client.calls) == 2


def test_task_cap_is_enforced(subagents, audit):
    big = TaskGraph.model_validate(
        {
            "goal": "g",
            "tasks": [
                {"id": f"t{i}", "subagent": "implementer", "objective": "o"} for i in range(5)
            ],
        }
    )
    client = FakePlannerClient(big, big)
    planner = Planner(client, subagents, max_tasks=3, run_id=None)

    with pytest.raises(PlanningError, match="the cap is 3"):
        planner.plan("do everything")


def test_planner_requests_structured_output(subagents):
    client = FakePlannerClient(graph())
    Planner(client, subagents).plan("fix login")
    assert client.calls[0][2]["schema"] is TaskGraph


def test_render_is_readable():
    out = graph().render()
    assert "impl (implementer)" in out
    assert "rev (reviewer) <- impl" in out


def test_task_defaults_are_permissive():
    t = Task(id="a", subagent="implementer", objective="o")
    assert t.depends_on == []
    assert t.inputs == {}
