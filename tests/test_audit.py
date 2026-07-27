import pytest

from conductor.audit import AuditStore, EventType, RunStatus


@pytest.fixture
def store(tmp_path):
    s = AuditStore(tmp_path / "test.db")
    yield s
    s.close()


def test_run_lifecycle(store):
    run_id = store.start_run("ship the thing")
    run = store.get_run(run_id)
    assert run["goal"] == "ship the thing"
    assert run["status"] == RunStatus.RUNNING

    store.finish_run(run_id, RunStatus.COMPLETED, summary="done")
    run = store.get_run(run_id)
    assert run["status"] == RunStatus.COMPLETED
    assert run["finished_at"] is not None


def test_events_are_ordered_and_appended(store):
    run_id = store.start_run("goal")
    store.append(run_id, actor="orchestrator", event_type=EventType.PLAN_CREATED, payload={"n": 3})
    store.append(run_id, actor="implementer", event_type=EventType.DELEGATION)

    events = store.events(run_id)
    assert [e.seq for e in events] == [1, 2, 3]
    assert events[0].event_type is EventType.RUN_STARTED
    assert events[1].payload == {"n": 3}


def test_tool_call_writes_ahead_then_records_result(store):
    run_id = store.start_run("goal")
    with store.tool_call(run_id, actor="reviewer", tool="gh.get_issue", arguments={"n": 7}) as out:
        out["result"] = {"title": "Broken login"}

    types = [e.event_type for e in store.events(run_id)]
    assert types == [EventType.RUN_STARTED, EventType.TOOL_CALL, EventType.TOOL_RESULT]

    call, result = store.events(run_id)[1], store.events(run_id)[2]
    assert result.parent_event_id == call.event_id
    assert result.payload["result"] == {"title": "Broken login"}


def test_tool_call_records_error_and_reraises(store):
    """The call event must survive the failure — that is the whole point of write-ahead."""
    run_id = store.start_run("goal")

    with pytest.raises(RuntimeError):
        with store.tool_call(run_id, actor="implementer", tool="gh.push", arguments={}) as out:
            out["result"] = None
            raise RuntimeError("network died")

    events = store.events(run_id)
    assert [e.event_type for e in events] == [
        EventType.RUN_STARTED,
        EventType.TOOL_CALL,
        EventType.ERROR,
    ]
    assert "network died" in events[2].payload["error"]
    assert events[2].parent_event_id == events[1].event_id


def test_cost_breakdown_sums_per_actor(store):
    run_id = store.start_run("goal")
    store.record_usage(
        run_id,
        actor="orchestrator",
        model="claude-opus-5",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    store.record_usage(
        run_id,
        actor="orchestrator",
        model="claude-opus-5",
        usage={"input_tokens": 20, "output_tokens": 5},
    )
    store.record_usage(
        run_id,
        actor="doc_writer",
        model="claude-sonnet-5",
        usage={"input_tokens": 7, "output_tokens": 3},
    )

    costs = store.cost_breakdown(run_id)
    assert costs["orchestrator"]["input_tokens"] == 120
    assert costs["orchestrator"]["output_tokens"] == 55
    assert costs["doc_writer"]["input_tokens"] == 7
