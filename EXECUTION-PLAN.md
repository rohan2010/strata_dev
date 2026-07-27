# Conductor — Execution Plan

Derived from `PRD-multi-agent-orchestrator-system (1).md` (Draft v0.1, 20 Jul 2026).
Plan date: 26 Jul 2026. Repo state at planning time: empty (PRD only, no commits).

**Stack decisions (confirmed):** Python 3.12 + FastAPI backend, LangGraph for the
orchestration graph, Next.js run view. Timeline re-scoped against PRD §12 rather than
taken as given — see §7 for what moved and why.

---

## 1. Architecture commitments

These are the decisions that are expensive to reverse. Make them in week 1.

| Decision | Choice | Why |
|---|---|---|
| Orchestration runtime | LangGraph `StateGraph`. **One generic `run_task` node, not one node per subagent** — the node resolves its manifest at dispatch time, so the graph has no branch to extend when a subagent is added (FR-S4). Dispatch loops per dependency layer | Task graph (FR-O2) is a first-class object rather than prompt-implied; checkpointer gives resumable runs (NFR reliability) for free |
| Subagent isolation | Each node owns its own `messages` list; only a structured summary is written back to graph state | FR-S1 / architectural note in §7 of the PRD. The orchestrator must never accumulate subagent transcripts |
| Subagent registration | YAML manifest per subagent (`role`, `route`, `effort`, `tools`, `prompt`, `risk_profile`), loaded at boot into a registry; graph nodes built from the registry | FR-S4. Cheap now, a rewrite later — build it in Phase 0 even though only two subagents exist |
| Model access | LangChain `init_chat_model` behind `conductor.models.ModelClient`; **no provider SDK is imported anywhere else** | Multi-provider from day one (Anthropic, OpenAI, Gemini, Ollama). The one thing a wrapper genuinely buys is cross-provider tool-schema and usage normalization, which is exactly what we need and would otherwise hand-write three times |
| Model routing | `models.yaml`: model catalogue + role -> ordered fallback chain. Manifests name a **route**, never a model id | Moving the Reviewer from Opus to Gemini is a one-line config edit. Also the only place cost and capability live, so `conductor models` can answer "what runs which job" |
| Policy engine | Standalone module, called by the graph's tool-dispatch wrapper. Agents never call tools directly | FR-H1 + §10. If the gate lives in a prompt, it isn't a gate |
| Persistence | Postgres (runs, entities, decisions, conventions, approvals, audit) + pgvector (semantic recall) | One datastore to operate. Avoids a separate vector DB until scale demands it |
| Connectors | MCP servers, reached from LangGraph tool nodes. Registry table the planner queries at plan time | FR-T1, FR-T4 |
| Audit | Every tool call, delegation, and approval written to an append-only `run_events` table before the action executes, not after | FR-A1. Write-ahead means a crash mid-tool-call is still traceable |

### Model selection (closes PRD §14 Q1)

**Revised 27 Jul 2026: multi-provider.** The original single-vendor table is now the set of
*defaults* in `models.yaml`, not a commitment. Roles route to an ordered fallback chain, and
which vendor serves a role is config.

| Role | Preferred | Fallback | Effort | Rationale |
|---|---|---|---|---|
| Orchestrator / planner | `claude-opus-5` | `gemini-pro` | `high`, `xhigh` for large task graphs | Long-horizon planning is where Opus 5 separates. $5/$25 per MTok |
| Implementer | `claude-opus-5` | `gpt-flagship` | `xhigh` | Coding is the beachhead's quality bar. Do not cheap out here |
| Reviewer | `gemini-pro` | `claude-opus-5` | `xhigh` | **Deliberately cross-vendor from the Implementer.** A reviewer sharing the author's model shares its blind spots; this is the cheapest independence we can buy |
| Tester/QA, Doc writer | `claude-sonnet-5` | `gemini-flash` | `medium` | Narrow, well-specified. $3/$15 ($2/$10 intro through 2026-08-31) |
| Synthesizer, Ticket writer | `claude-sonnet-5` | `gemini-pro` | `high` | Judgment-bearing but bounded |
| Classification, routing heuristics | `local-tiny` (Ollama) | `claude-haiku-4-5` | n/a | Local and free for genuinely mechanical steps; hosted fallback covers "Ollama isn't running" |
| Transcript cleanup | `local-small` (Ollama) | `claude-haiku-4-5` | n/a | High volume, low judgement — the case where local models pay for themselves |

**Effort is declared intent, not a provider parameter.** Providers disagree about the knob
and its range: Anthropic accepts `low..max`, OpenAI and Gemini top out at `high`, and Ollama
has an on/off switch rather than a scale. `conductor.models.providers` translates, and always
clamps *downward* — effort is a cost lever as well as a quality one, so we never silently buy
more reasoning than was asked for. `conductor models effort` prints declared-vs-delivered per
subagent, which is the answer to "why did that fallback run feel shallower".

**Cost is per-provider and may be unknown.** `models.yaml` carries prices per model, and a
model with no price still runs — its usage reports tokens with no dollars, and
`cost_breakdown` returns an `unpriced_calls` count alongside the total. A silent $0 in a cost
dashboard is worse than an honest gap.

Three things that follow from mixed models and are easy to get wrong:

- **Prompt caches are model-scoped.** Never switch models mid-conversation to save cost —
  it invalidates the whole prefix. The cheap-model path is *spawn a subagent on the cheaper
  model*, which is what the architecture already does. This is a reason the isolation
  boundary matters commercially, not just for context hygiene.
- **Thinking is on by default on Opus 5** (unlike 4.8). `max_tokens` caps thinking + text
  together, so size it generously — 64K when running `xhigh`.
- **Handle refusals before reading content, and do not retry them elsewhere.** A refusal is
  a normal HTTP 200 with empty or partial content, spelled differently per vendor
  (`stop_reason: "refusal"`, `finish_reason: "content_filter"`, …). `ModelClient` normalizes
  the vocabulary and raises `ModelRefusal` — which explicitly **abandons the fallback chain**.
  A decline is a judgement about the request, not an outage; rerouting it to vendor after
  vendor until one complies is a safety-bypass pattern, not resilience. Transport errors,
  rate limits, and outages *do* fall through to the next candidate, and each failed attempt
  writes its own ERROR event.

Cost control (NFR): a hard output-token ceiling per subagent (`max_tokens` in the manifest,
translated to whatever the provider calls it), plus per-call `usage` logged into `run_events`
with the model alias, provider, and computed dollar cost — FR-A3 falls out of this for free.
Provider-native budget features (Anthropic's `output_config.task_budget`) are a Phase 1
refinement layered on top, not the primary control, since they do not exist on every vendor.

### Prompt-tuning notes for Opus 5 (bake into system prompts from the start)

The model's defaults fight several things this product needs. Cheaper to encode now than to
debug as "the agent is being weird" in week 6.

**Caveat now that routes are multi-vendor:** these notes are tuned to Opus 5's defaults, and
the shipped prompts are written against them. A fallback to GPT or Gemini runs the same
prompt on a model with different defaults — so subagent prompts should state the behaviour
they want rather than lean on a vendor's disposition. This is a known gap: the eval set
(§6) needs to run per route candidate, not just the preferred one.

- It **verifies its own work unprompted** — do *not* add "double-check your output"
  instructions or a separate verification subagent. That causes over-verification with no
  quality gain.
- It **delegates readily** — the orchestrator prompt needs an explicit subagent cap, not
  encouragement to delegate.
- It **expands scope** and **writes long deliverables** — add explicit scope-discipline and
  length instructions to Implementer and Doc writer.

---

## 2. Phase 0 — Spike (weeks 1–2)

**Goal:** prove plan → delegate → aggregate on one real GitHub issue. No memory, no policy
engine, approvals are a blocking CLI prompt.

| # | Task | Output |
|---|---|---|
| 0.1 | Repo scaffold: `uv` project, FastAPI app, Postgres via docker-compose, Alembic, ruff + pytest + CI | `git init` → first commit |
| 0.2 | `run_events` table + writer. Everything downstream logs through it | Audit spine exists before the first agent does |
| 0.3 | Subagent manifest schema + registry loader | `subagents/implementer.yaml`, `subagents/reviewer.yaml` |
| 0.3b | Model routing registry + provider-agnostic client (`models.yaml`, `conductor.models`) | Anthropic/OpenAI/Gemini/Ollama behind one call, with fallback chains and per-provider cost |
| 0.4 | LangGraph skeleton: plan → dispatch → aggregate, with dispatch as a **loop hub** so the graph runs one dependency layer per superstep. SQLite checkpointer (Postgres in Phase 1) | Resumable runs from day one; parallel dispatch (FR-S3) falls out of the layering rather than waiting for Phase 1 |
| 0.5 | Planner: goal → typed task graph (Pydantic, cross-provider structured output), roster and connectors rendered from the registries, one repair pass on an invalid plan | FR-O2 |
| 0.6 | GitHub MCP connector wired to Implementer (write) and Reviewer (read-only) | FR-T2 least-privilege proven with two scopes |
| 0.7 | CLI: `conductor run "<goal>" --repo owner/name`, plus `models` / `doctor` inspection. **Approval gating not yet present** | FR-O6 in its crudest form |
| 0.8 | **Baseline instrumentation:** manually time 10 real issue→PR cycles, record in a sheet | Without this, "≥50% time reduction" (§11) is unmeasurable forever |

**Exit criterion:** one real GitHub issue → open PR with a review comment, fully traced in
`run_events`, without hand-editing state mid-run.

**Status (27 Jul 2026):** 0.2–0.7 built and covered by tests; the run path is exercised end
to end against scripted models, not yet against a live provider. Still open in Phase 0:

- **0.1** — CI and docker-compose Postgres are not set up; the repo has no commits yet.
- **0.8** — the manual baseline has not been recorded. It gets unrecoverably harder the
  longer the tool is in use, so it should happen before the first real run, not after.
- The exit criterion itself is unmet: no run has touched a live provider or a real repo,
  because `models.yaml` still carries unverified non-Anthropic ids and no keys are set.

**Risk to watch:** if the planner produces unusable task graphs on 3 of 5 trial goals, stop
and fix planning before adding subagents. Everything downstream compounds on it.

---

## 3. Phase 1 — MVP: Ticket → PR (weeks 3–7)

*One week longer than PRD §12's wk 3–6. See §7.*

| # | Task | Requirement |
|---|---|---|
| 1.1 | Policy engine: risk classifier (`auto` / `notify` / `gate`), default-deny list (merge, deploy, delete, external message, spend), per-action-type thresholds in config | FR-H1, FR-H4, FR-H5, §10 |
| 1.2 | Tool-dispatch wrapper: every MCP call routes through policy → audit → execute | The enforcement point. Nothing bypasses it |
| 1.3 | Approval card model (what / why / expected effect / reversibility) + resume-from-approval | FR-H2 |
| 1.4 | Tester/QA + Doc writer subagents (manifest-only additions — no orchestrator changes; this is the FR-S4 acceptance test) | FR-S4 |
| 1.5 | Parallel dispatch of independent graph branches | FR-S3, NFR performance |
| 1.6 | Failure containment: per-subagent timeout, partial-result aggregation, structured failure reason | FR-O7, NFR reliability |
| 1.7 | Structured memory store only: runs, decisions, conventions, approvals. **No vector store yet** | Partial FR-M2 |
| 1.8 | Convention memory read into subagent prompts at dispatch ("we use trunk-based dev", "always add tests") | FR-M3, and the first thing that makes month two better than month one |
| 1.9 | Run view (Next.js): task graph, per-subagent actions, artifacts, token/cost per subagent | FR-C3, FR-A2, FR-A3 |
| 1.10 | Web chat surface for goals + approvals | FR-C1 |

**Exit criterion:** five consecutive real tickets go issue → reviewed, tested, documented PR
with no manual intervention except approvals. Measure against the 0.8 baseline.

---

## 4. Phase 2 — v1: memory + Insight → Ticket (weeks 8–13)

| # | Task | Requirement |
|---|---|---|
| 2.1 | pgvector store + embedding pipeline: docs, findings, code chunks, past readouts | FR-M2 |
| 2.2 | Memory query tool available to all agents + a user-facing "what did we decide about auth in March?" query path | FR-M4 |
| 2.3 | Memory view/edit/delete UI, provenance on every stored decision | FR-M5, mitigates the memory-poisoning risk in §13 |
| 2.4 | Slack connector: approval cards, notifications, status replies | FR-C2 |
| 2.5 | Synthesizer subagent: findings → themes with evidence links | Partial beachhead workflow 3 |
| 2.6 | Ticket writer subagent: finding → scoped issue (severity, segment, repro) | Beachhead workflow 1 |
| 2.7 | Insight → Ticket → PR chained end to end on one real usability finding | §11 "beachhead proof" |
| 2.8 | Cost dashboard + per-run budget enforcement with alerting | NFR cost control |
| 2.9 | Daily dogfood; weekly review of escalation/failure log | §11 |

**Exit criterion:** the loop closes on a real finding, and the team uses it daily without
being asked to.

---

## 5. Deferred to post-v1 (recommended cut — see §7)

- Transcriber subagent + transcription connector
- Participant recruiting connector, calendar integration, participant contact flows
- Automated readout drafting
- Figma connector (live or export)
- Threshold personalization, multi-user, new-domain subagents (PRD §12 Phase 3)

The Synthesizer ships in Phase 2 taking **manually supplied transcripts**. That preserves
the synthesis value without pulling in consent handling, PII minimization in logs, and a
participant-contact approval path — each of which is a real workstream, not a checkbox.

---

## 6. Cross-cutting workstreams

**Security / privacy (continuous, starts Phase 0):** secrets in a vault from the first
connector, never in prompts. Per-subagent MCP scopes enforced in the manifest, not by
convention. PII redaction in `run_events` before Phase 2 touches any research data.

**Evals (starts end of Phase 0, run every phase):** a fixed set of 15 goals — 5 planning,
5 implementation, 5 memory-recall — run on every orchestrator prompt change. Without this,
the §11 metrics are anecdote. This is also how you tune `effort` per role instead of
guessing: sweep `medium`/`high`/`xhigh` on the eval set and pick per subagent.

**Prompt-injection surface (add in Phase 1):** the Implementer reads issue text written by
outsiders and acts on it with write scopes. Treat all connector-sourced content as untrusted
input to the policy engine, not as instructions. The PRD does not mention this risk; it
belongs in §13.

---

## 7. Where this plan departs from PRD §12, and why

The PRD's phasing is directionally right but packs roughly 16 weeks of work into 12. Four
changes:

1. **MVP gets a fifth week (wk 3–7).** The policy engine (FR-H1–H5) is specified as one
   bullet in Phase 1 but is a genuine subsystem — classifier, thresholds, approval cards,
   resume semantics, audit integration. It is also the thing that makes the product safe to
   dogfood, so it cannot be thinned.

2. **Vector memory moves from Phase 1 to Phase 2.** "Basic long-term memory" in the PRD's
   Phase 1 conflates two different builds. Structured memory (conventions, decisions) is
   what makes Ticket→PR better and is cheap; semantic recall is a pipeline with its own
   quality bar and eval needs. Splitting them lets Phase 1 ship on time and gives Phase 2 a
   coherent memory theme.

3. **"Study end-to-end" comes out of v1.** It is a second product surface: recruiting,
   transcription, consent-scoped storage, participant contact gating. The PRD's own §14
   flags recruiting as unresolved (Q5) and Figma as unresolved (Q4) — both sit in this
   workflow. Shipping Insight→Ticket→PR deeply is a stronger v1 than shipping three
   workflows shallowly, and the beachhead argument in the scope note supports depth.

4. **Baseline measurement is added to Phase 0.** §11 targets a ≥50% time reduction against
   manual, and there is currently no manual baseline recorded anywhere. Ten timed cycles in
   week 1 costs a few hours and is unrecoverable later.

**If the 12-week date is fixed and non-negotiable,** the cut order is: (a) Slack approvals
→ web-only, (b) Synthesizer → Phase 3, (c) memory view/edit UI → API-only with a CLI. Do not
cut the policy engine, the audit log, or evals; those are what make the rest trustworthy.

---

## 8. Open questions still requiring a decision

Resolved by this plan: §14 Q1 (models — see §1; now "which route", not "which vendor"),
Q2 (LangGraph), Q4/Q5 (deferred with the study workflow).

Still open, needed by the date shown:

| Question | Needed by |
|---|---|
| Verify the placeholder OpenAI/Gemini model ids and prices in `models.yaml`, and `ollama pull` the local models — everything non-Anthropic there is structurally correct but unconfirmed | Before the first multi-provider run |
| Per-provider prompt divergence: do fallback candidates need their own prompt variants, or is one prompt written to state its requirements explicitly enough? Answered by running the eval set per candidate | Phase 1, task 1.4 |
| Auto-written vs. curated memory (§14 Q3) — proposal: auto-write runs and decisions, require explicit confirmation for conventions, since a wrong convention poisons every future run | Start of Phase 2 |
| Self-hosted vs. Anthropic-hosted execution for the Implementer's test runs | Phase 1, task 1.4 |
| Issue tracker: GitHub Issues only, or Jira too (FR-T3 lists both) | Phase 2, task 2.6 |

---

## 9. First five commits

1. Repo scaffold, CI, docker-compose Postgres
2. `run_events` schema + writer + test
3. Subagent manifest schema + registry loader
4. LangGraph skeleton with a stub planner and one echo subagent
5. GitHub MCP connector with read-only Reviewer scope
