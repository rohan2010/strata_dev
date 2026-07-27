import pytest
from pydantic import ValidationError

from conductor.audit import AuditStore, EventType
from conductor.models import (
    AllCandidatesFailed,
    Capability,
    Effort,
    ModelClient,
    ModelRefusal,
    ModelRegistry,
    Provider,
    clamp_effort,
)
from conductor.models.client import _normalize_usage, _refusal_reason, _text_of
from conductor.models.providers import build_model_kwargs

RAW = {
    "models": {
        "opus": {
            "provider": "anthropic",
            "id": "claude-opus-5",
            "supports": ["tools", "structured_output", "thinking"],
            "price": {"input": 5.0, "output": 25.0},
        },
        "gemini": {
            "provider": "google_genai",
            "id": "gemini-3-pro",
            "supports": ["tools", "structured_output", "thinking"],
        },
        "tiny": {
            "provider": "ollama",
            "id": "llama3.2:3b",
            "supports": [],
            "price": {"input": 0.0, "output": 0.0},
        },
    },
    "routes": {"reviewer": ["gemini", "opus"], "classifier": ["tiny", "opus"]},
    "default_route": "reviewer",
}


@pytest.fixture
def registry():
    return ModelRegistry.model_validate(RAW)


# ---- registry ------------------------------------------------------------


def test_route_resolves_to_ordered_chain(registry):
    assert [m.alias for m in registry.chain("reviewer")] == ["gemini", "opus"]
    assert registry.primary("reviewer").provider is Provider.GOOGLE


def test_route_naming_unknown_model_fails_at_load():
    bad = RAW | {"routes": {"reviewer": ["nope"]}, "default_route": None}
    with pytest.raises(ValidationError, match="unknown model"):
        ModelRegistry.model_validate(bad)


def test_empty_fallback_chain_rejected():
    bad = RAW | {"routes": {"reviewer": []}, "default_route": None}
    with pytest.raises(ValidationError, match="empty fallback chain"):
        ModelRegistry.model_validate(bad)


def test_capability_requirement_filters_candidates(registry):
    """A model that cannot do structured output must not be handed a typed request."""
    chain = registry.chain("classifier", requires={Capability.STRUCTURED_OUTPUT})
    assert [m.alias for m in chain] == ["opus"]


def test_no_capable_candidate_is_a_clear_error(registry):
    with pytest.raises(ValueError, match="no model on route"):
        registry.chain("classifier", requires={Capability.VISION})


def test_missing_route_falls_back_to_default(registry):
    assert [m.alias for m in registry.chain("brand_new_role")] == ["gemini", "opus"]


def test_missing_route_without_default_raises():
    reg = ModelRegistry.model_validate(RAW | {"default_route": None})
    with pytest.raises(KeyError, match="no route for role"):
        reg.chain("brand_new_role")


def test_unpriced_models_are_reported_not_treated_as_free(registry):
    assert registry.unpriced() == ["gemini"]


def test_describe_shows_every_route_and_model(registry):
    out = registry.describe()
    for name in ["reviewer", "classifier", "opus", "gemini", "tiny"]:
        assert name in out


def test_shipped_registry_loads():
    registry = ModelRegistry.from_file("models.yaml")
    assert registry.default_route in registry.routes
    assert "opus-5" in registry.models


# ---- pricing -------------------------------------------------------------


def test_cost_is_per_million_tokens(registry):
    price = registry.models["opus"].price
    assert price is not None
    # 1M in @ $5 + 1M out @ $25
    assert price.cost_usd(input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(30.0)


def test_cache_reads_billed_at_their_own_rate_when_declared():
    reg = ModelRegistry.model_validate(
        RAW
        | {
            "models": RAW["models"]
            | {
                "opus": {
                    "provider": "anthropic",
                    "id": "claude-opus-5",
                    "supports": ["tools"],
                    "price": {"input": 5.0, "output": 25.0, "cache_read": 0.5},
                }
            }
        }
    )
    price = reg.models["opus"].price
    assert price is not None
    # 1M input of which 800K was a cache hit: 200K @ $5 + 800K @ $0.50 + 0 out
    assert price.cost_usd(
        input_tokens=1_000_000, output_tokens=0, cache_read=800_000
    ) == pytest.approx(1.0 + 0.4)


# ---- provider translation -------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "requested", "expected"),
    [
        (Provider.ANTHROPIC, Effort.XHIGH, Effort.XHIGH),
        (Provider.ANTHROPIC, Effort.MAX, Effort.MAX),
        (Provider.OPENAI, Effort.XHIGH, Effort.HIGH),
        (Provider.OPENAI, Effort.MEDIUM, Effort.MEDIUM),
        (Provider.GOOGLE, Effort.MAX, Effort.HIGH),
        (Provider.OLLAMA, Effort.MAX, None),
    ],
)
def test_effort_clamps_downward_never_up(provider, requested, expected):
    assert clamp_effort(provider, requested) == expected


def test_max_tokens_param_name_differs_per_provider():
    common = {"effort": Effort.HIGH, "max_tokens": 4096, "supports_thinking": True}
    assert build_model_kwargs(Provider.ANTHROPIC, **common)["max_tokens"] == 4096
    assert build_model_kwargs(Provider.GOOGLE, **common)["max_output_tokens"] == 4096
    assert build_model_kwargs(Provider.OLLAMA, **common)["num_predict"] == 4096


def test_ollama_gets_a_reasoning_switch_not_a_scale():
    on = build_model_kwargs(
        Provider.OLLAMA, effort=Effort.XHIGH, max_tokens=None, supports_thinking=True
    )
    off = build_model_kwargs(
        Provider.OLLAMA, effort=Effort.LOW, max_tokens=None, supports_thinking=True
    )
    assert on == {"reasoning": True}
    assert off == {"reasoning": False}
    assert "reasoning_effort" not in on


def test_non_thinking_model_gets_no_effort_param():
    kwargs = build_model_kwargs(
        Provider.ANTHROPIC, effort=Effort.MAX, max_tokens=None, supports_thinking=False
    )
    assert kwargs == {}


def test_per_model_params_override_the_translation():
    kwargs = build_model_kwargs(
        Provider.GOOGLE,
        effort=Effort.HIGH,
        max_tokens=100,
        supports_thinking=True,
        extra={"max_output_tokens": 999, "safety_settings": {"x": "y"}},
    )
    assert kwargs["max_output_tokens"] == 999
    assert kwargs["safety_settings"] == {"x": "y"}


# ---- client behaviour -----------------------------------------------------


class FakeMessage:
    def __init__(self, text="ok", usage=None, metadata=None):
        self.content = text
        self.usage_metadata = usage
        self.response_metadata = metadata or {}


class FakeModel:
    """Stands in for a chat model. `error` makes the candidate fail."""

    def __init__(self, message=None, error=None):
        self.message = message or FakeMessage()
        self.error = error
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.message


@pytest.fixture
def client(registry, tmp_path):
    audit = AuditStore(tmp_path / "t.db")
    c = ModelClient(registry, audit=audit, run_id=audit.start_run("goal"))
    yield c, audit
    audit.close()


def _stub(client, mapping):
    """Pin each alias to a fake model, bypassing real provider construction."""
    client.build = lambda spec, **_: mapping[spec.alias]


def test_falls_back_to_next_candidate_and_records_why(client):
    c, audit = client
    first = FakeModel(error=RuntimeError("503 upstream"))
    second = FakeModel(FakeMessage("reviewed"))
    _stub(c, {"gemini": first, "opus": second})

    result = c.invoke("reviewer", "hi", actor="reviewer")

    assert result.text == "reviewed"
    assert result.spec.alias == "opus"
    assert result.attempts == ["gemini", "opus"]
    assert result.fell_back

    errors = [e for e in audit.events(c._run_id) if e.event_type is EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].payload["model"] == "gemini"
    assert errors[0].payload["falling_back_to"] == "opus"
    assert "503 upstream" in errors[0].payload["error"]


def test_refusal_is_not_shopped_to_another_provider(client):
    """A decline is a judgement about the request, not an outage."""
    c, _ = client
    refusing = FakeModel(FakeMessage("", metadata={"stop_reason": "refusal"}))
    other = FakeModel(FakeMessage("would have complied"))
    _stub(c, {"gemini": refusing, "opus": other})

    with pytest.raises(ModelRefusal):
        c.invoke("reviewer", "hi")

    assert other.calls == 0


def test_all_candidates_failing_reports_each_one(client):
    c, _ = client
    _stub(
        c,
        {
            "gemini": FakeModel(error=RuntimeError("down")),
            "opus": FakeModel(error=RuntimeError("also down")),
        },
    )
    with pytest.raises(AllCandidatesFailed) as exc:
        c.invoke("reviewer", "hi")
    assert [alias for alias, _ in exc.value.failures] == ["gemini", "opus"]


def test_usage_and_cost_land_in_the_audit_log(client):
    c, audit = client
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "total_tokens": 2_000_000,
        "input_token_details": {"cache_read": 0},
    }
    _stub(
        c,
        {
            "gemini": FakeModel(error=RuntimeError("skip")),
            "opus": FakeModel(FakeMessage("x", usage=usage)),
        },
    )

    result = c.invoke("reviewer", "hi", actor="reviewer")

    assert result.cost_usd == pytest.approx(30.0)
    breakdown = audit.cost_breakdown(c._run_id)
    assert breakdown["reviewer"]["input_tokens"] == 1_000_000
    assert breakdown["reviewer"]["cost_usd"] == pytest.approx(30.0)
    assert breakdown["reviewer"]["unpriced_calls"] == 0


def test_unpriced_model_reports_tokens_but_no_dollars(client):
    """An unpriced model must not silently contribute $0 to a cost total."""
    c, audit = client
    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    _stub(c, {"gemini": FakeModel(FakeMessage("x", usage=usage)), "opus": FakeModel()})

    result = c.invoke("reviewer", "hi", actor="reviewer")

    assert result.cost_usd is None
    total, unpriced = audit.run_cost_usd(c._run_id)
    assert total == 0.0
    assert unpriced == 1
    assert audit.cost_breakdown(c._run_id)["reviewer"]["input_tokens"] == 100


def test_structured_request_skips_incapable_candidates(client, registry):
    """`tiny` declares no structured_output, so a typed call must not reach it."""
    c, _ = client
    chain = registry.chain("classifier", requires={Capability.STRUCTURED_OUTPUT})
    assert "tiny" not in [m.alias for m in chain]


# ---- normalizers ----------------------------------------------------------


def test_usage_normalizer_flattens_nested_details():
    msg = FakeMessage(
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 4, "cache_creation": 1},
            "output_token_details": {"reasoning": 3},
        }
    )
    assert _normalize_usage(msg) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cache_read": 4,
        "cache_creation": 1,
        "reasoning_tokens": 3,
    }


def test_usage_normalizer_tolerates_a_provider_reporting_nothing():
    assert _normalize_usage(FakeMessage())["input_tokens"] == 0


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"stop_reason": "refusal"}, "refusal"),
        ({"finish_reason": "content_filter"}, "content_filter"),
        ({"stop_reason": "end_turn"}, None),
        ({}, None),
    ],
)
def test_refusal_detection_spans_provider_vocabularies(metadata, expected):
    assert _refusal_reason(FakeMessage(metadata=metadata)) == expected


def test_text_extraction_handles_content_blocks():
    msg = FakeMessage(
        text=[
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
    )
    assert _text_of(msg) == "hello world"
