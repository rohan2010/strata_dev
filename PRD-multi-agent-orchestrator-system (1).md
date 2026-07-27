# PRD — Multi-Agent Orchestration System ("Conductor")

| Field | Value |
|---|---|
| **Document status** | Draft v0.1 — for review |
| **Owner** | [You] |
| **Contributors** | Eng, Design/Research |
| **Last updated** | 20 Jul 2026 |
| **Target release** | MVP in ~6 weeks, v1 in ~12 weeks (indicative) |

> **Scope decision (read first).** This PRD specifies a *general* orchestrator + specialist-subagent platform, but deliberately picks a **single beachhead for v1: the developer + UX-research workflow bridge** (turn research insights into shipped code, and close the loop back). We build the platform thin and the beachhead deep, dogfood it on our own work, then extend to other domains. If the beachhead should be different, change Section 4 and the rest still holds.

---

## 1. Summary

Knowledge workers spend most of their day *coordinating tools* rather than doing the work only they can do — copy-pasting between a code host, a design tool, a research repository, a ticket tracker, and five browser tabs. "Conductor" is a system where a single **generalist orchestrator agent** takes a goal in plain language, plans it, delegates to **specialist subagents**, and returns an outcome — pausing for human approval on the decisions that carry real risk. It has **compounding memory** (it learns your codebase, conventions, and past research) and connects to your existing tools once, so you talk in one place instead of ten tabs.

The pattern is domain-agnostic. v1 proves it on the workflow closest to our own pain: **the seam between UX research and engineering.**

---

## 2. Background & problem

Two observations motivate this product:

1. **Single agents hit a ceiling.** A lone LLM agent handling a large, multi-step job accumulates context bloat, loses focus, and can't parallelize. Splitting work across specialists — each with a narrow system prompt, its own tools, and isolated context — improves reliability and lets independent subtasks run at once.
2. **The expensive handoffs are between disciplines.** For a person who does both research and engineering, the costly gaps are: insight → requirement, requirement → implementation, and shipped feature → back to discovery. These handoffs are manual, lossy, and rarely closed.

Existing point tools (a coding copilot here, a research-analysis tool there) each solve one slice and leave the human owning every handoff. No single owner is accountable for the *outcome*.

### Problem statement
> A practitioner who spans research and engineering has no single system that can own a goal end to end across both disciplines, remember context over time, and act across their tools while keeping them in control of the high-judgment calls.

---

## 3. Goals & non-goals

### Goals
- **G1.** Let a user hand over a goal in natural language and receive a completed outcome, not a task list.
- **G2.** Route work to the right specialist automatically; the user never picks a subagent.
- **G3.** Keep the human in control via risk-scaled approval gates.
- **G4.** Accumulate durable memory so month two outperforms month one.
- **G5.** Integrate with existing tools via a standard protocol (MCP), connect-once.
- **G6.** Full auditability — every plan, delegation, tool call, and approval is traceable.

### Non-goals (for v1)
- **N1.** Not building new foundation models; we orchestrate existing ones.
- **N2.** Not a no-code workflow builder — users give goals, not flowcharts.
- **N3.** Not multi-tenant SaaS at launch; single-team, self-hosted/dogfood first.
- **N4.** Not autonomous action on irreversible/high-risk operations without approval.
- **N5.** Not replacing human judgment on architecture or research insight validation.

---

## 4. Target users & v1 beachhead

### Primary persona
**"Riya" — the builder-researcher.** Ships code and runs UX research. Owns the whole loop from "what should we build" to "did it work." Time is lost to transcription, boilerplate, test-writing, ticket-grooming, and synthesis busywork.

### v1 beachhead scope — three flagship workflows
1. **Insight → Ticket.** A usability finding becomes a prioritized, scoped issue (severity, affected segment, repro).
2. **Ticket → PR.** An issue is implemented, reviewed, tested, and documented into an open PR for human merge.
3. **Study end-to-end.** A research study is recruited, transcribed, synthesized into themes with evidence, and drafted into a readout.

Everything past v1 (other domains, other specialists) is roadmap, Section 12.

---

## 5. User stories

- As Riya, I want to **hand over a GitHub issue** and get back a reviewed, tested PR, so I only spend time on the architecture decision and the merge.
- As Riya, I want **session recordings turned into themed findings with clips**, so I skip manual transcription and tagging.
- As Riya, I want a **usability finding auto-drafted as a scoped ticket**, so insights don't die in a doc.
- As Riya, I want the system to **ask before anything irreversible** (merging, deleting, emailing a participant), so I stay in control.
- As Riya, I want the system to **remember our code conventions and past studies**, so I don't re-explain context every time.
- As Riya, I want to **see exactly what each subagent did**, so I can trust and debug the result.

---

## 6. Functional requirements

### 6.1 Orchestrator (generalist)
- **FR-O1.** Accept a goal in natural language on a supported channel.
- **FR-O2.** Decompose the goal into a task graph (dependencies + parallelizable branches).
- **FR-O3.** Select and dispatch subagents per task; never require the user to choose.
- **FR-O4.** Aggregate subagent results, resolve conflicts, and return a single coherent outcome.
- **FR-O5.** Detect when a step crosses an approval threshold and pause for the human.
- **FR-O6.** Maintain a working summary of progress the user can query at any time ("what's the status?").
- **FR-O7.** Fail gracefully — surface partial results and a clear reason on any subagent failure.

### 6.2 Specialist subagents
- **FR-S1.** Each subagent runs with a scoped system prompt, an allow-listed tool subset, and isolated context.
- **FR-S2.** Subagents return structured results (status, artifact refs, confidence, notes) — not free text only.
- **FR-S3.** Independent subagents run concurrently.
- **FR-S4.** New subagent types register via a manifest (role, tools, prompt, risk profile) without changing the orchestrator core.

**v1 subagent set**

| Subagent | Domain | Key tools |
|---|---|---|
| Implementer | Write/modify code | Code host, file I/O, run tests |
| Reviewer | Critique diffs | Code host (read), linters |
| Tester/QA | Author + run tests | Test runner, CI |
| Doc writer | Update docs/READMEs | File I/O, code host |
| Transcriber | Sessions → text | Transcription service, storage |
| Synthesizer | Cluster findings into themes | Research repo, vector store |
| Ticket writer | Findings → scoped issues | Issue tracker |

### 6.3 Memory
- **FR-M1.** Short-term working memory scoped to a run.
- **FR-M2.** Long-term memory: a **vector store** (semantic recall of docs, findings, code chunks) plus a **structured store** (entities, decisions, conventions, approvals).
- **FR-M3.** Every completed run writes durable memory (what was done, why, what was decided).
- **FR-M4.** Memory is queryable by any agent and by the user ("what did we decide about auth in March?").
- **FR-M5.** User can view, correct, and delete memory (trust + privacy).

### 6.4 Tool / connector layer
- **FR-T1.** Integrate external systems via **MCP** connectors; connect-once, reuse across agents.
- **FR-T2.** Per-connector scoped permissions (least privilege) and secure secret storage.
- **FR-T3.** v1 connectors: code host (GitHub), issue tracker (GitHub Issues/Jira), design/research source (Figma export or research repo), transcription, calendar.
- **FR-T4.** A connector registry the orchestrator queries to discover available capabilities at plan time.

### 6.5 Human-in-the-loop (HITL) approval
- **FR-H1.** A policy engine classifies each proposed action by risk: **auto** (reversible, low stakes), **notify** (proceed but inform), **gate** (block until approved).
- **FR-H2.** Gated actions surface an approval card (what, why, expected effect, reversibility) on the user's channel.
- **FR-H3.** Approvals, holds, and edits are logged and become memory.
- **FR-H4.** Default-deny for anything irreversible: merge, deploy, delete, external email/message, spend.
- **FR-H5.** Users can adjust thresholds per action type over time.

### 6.6 Channels / interface
- **FR-C1.** Primary chat interface (web) for goals, status, and approvals.
- **FR-C2.** At least one push channel for approvals/notifications (Slack for v1).
- **FR-C3.** A run view showing the task graph, each subagent's actions, and artifacts produced.

### 6.7 Observability & audit
- **FR-A1.** Trace every run: plan, each delegation, every tool call (inputs/outputs), every approval.
- **FR-A2.** Traces are inspectable in the run view and exportable.
- **FR-A3.** Token/cost accounting per run and per subagent.

---

## 7. System architecture (high level)

```
                ┌───────────────────────────────────────────────┐
   User ──────► │  Channel layer (web chat, Slack)              │
                └───────────────┬───────────────────────────────┘
                                │  goal / approvals
                ┌───────────────▼───────────────┐
                │      Orchestrator agent        │  plan → route → aggregate
                │  (planner + router + HITL)     │
                └───┬───────────┬───────────┬────┘
        dispatch    │           │           │   (parallel where independent)
             ┌──────▼──┐  ┌─────▼───┐  ┌────▼─────┐
             │Subagent │  │Subagent │  │Subagent  │  scoped prompt + tools + context
             │  A      │  │  B      │  │  C ...   │
             └────┬────┘  └────┬────┘  └────┬─────┘
                  │            │            │
        ┌─────────▼────────────▼────────────▼─────────┐
        │      Tool / connector layer (MCP)            │  GitHub, Jira, Figma, ...
        └─────────┬────────────────────────┬───────────┘
                  │                         │
        ┌─────────▼────────┐      ┌─────────▼──────────┐
        │  Memory layer    │      │  HITL policy engine │
        │ vector + struct. │      │  risk classify/gate │
        └──────────────────┘      └────────────────────┘
                  │
        ┌─────────▼────────┐
        │  Observability / audit log (traces, cost)    │
        └──────────────────────────────────────────────┘
```

**Key architectural decisions**
- **Isolated subagent context** prevents cross-contamination and context bloat; the orchestrator holds only summaries.
- **MCP as the integration standard** so connectors are reusable and the ecosystem is open.
- **Policy engine separate from agents** so safety rules aren't at the mercy of a prompt.
- **Everything writes to the audit log** — non-negotiable for trust in a multi-agent system.

---

## 8. Non-functional requirements

- **Security:** least-privilege connector scopes; secrets in a vault, never in prompts; no connector access an agent doesn't need for its role.
- **Privacy:** user-owned memory with view/delete; research participant data handled per consent; PII minimization in logs.
- **Reliability:** subagent failure is contained (partial results returned); runs are resumable.
- **Performance:** parallelize independent subtasks; target median simple-goal completion under a few minutes.
- **Cost control:** per-run token budgets and cost accounting; cheaper models for narrow subagents where quality allows.
- **Auditability:** 100% of tool calls and approvals traced.
- **Extensibility:** add a subagent or connector without touching the orchestrator core.

---

## 9. Data & memory model (sketch)

- **Entity store:** projects, people/participants, decisions, conventions, connectors.
- **Run store:** goal, plan/task-graph, subagent actions, artifacts, approvals, cost.
- **Vector store:** embeddings of docs, research findings, code chunks, past readouts — for semantic recall.
- **Convention memory:** explicit rules the system learned or was told ("we use trunk-based dev," "always add tests," "never DM participants without approval").

---

## 10. Safety & permissions policy

- Default-deny on irreversible actions (Section 6.5).
- Every subagent gets the **minimum** connector scopes for its role (e.g., Reviewer = read-only on the code host).
- Approval decisions are durable and auditable.
- The policy engine is the single source of truth for what needs a gate — agents cannot override it.

---

## 11. Success metrics

| Metric | What it tells us | v1 target (indicative) |
|---|---|---|
| **Task completion rate (no human correction)** | Core reliability | ≥ 70% of goals |
| **Human-approval overhead** | How often we interrupt | Gated on < 30% of steps, mostly the right ones |
| **Time-to-outcome vs. manual** | Value delivered | ≥ 50% reduction on beachhead workflows |
| **Memory recall accuracy** | Is memory useful | ≥ 90% correct on "what did we decide" queries |
| **Escalation/failure rate** | Where it breaks | Trending down week over week |
| **Insight→ticket→PR loop closure** | Beachhead proof | End-to-end demo on a real finding |
| **Weekly active dogfood use** | Do *we* rely on it | Daily use by the team |

---

## 12. Roadmap / phasing

- **Phase 0 — Spike (wk 1–2):** orchestrator + two subagents (Implementer, Reviewer) on a single connector (GitHub), no memory, manual approval. Prove the plan→delegate→aggregate loop.
- **Phase 1 — MVP (wk 3–6):** add Tester, Doc writer; MCP connector layer; basic long-term memory; HITL policy engine; run view. Ships the **Ticket → PR** workflow.
- **Phase 2 — v1 (wk 7–12):** add research subagents (Transcriber, Synthesizer, Ticket writer); Slack approvals; compounding memory; the full **Insight → Ticket** and **Study end-to-end** loops. Dogfood daily.
- **Phase 3 — Extend (post-v1):** new domains/subagents via manifest; threshold personalization; cost optimization; consider multi-user.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Multi-agent coordination is unreliable / agents loop | Start with 2 subagents; strict task graphs; timeouts + failure containment |
| Context bloat in the orchestrator | Orchestrator holds summaries only; full context lives in subagents |
| Over- or under-gating (annoying vs. unsafe) | Tunable thresholds; measure approval overhead; default-deny irreversible only |
| Connector/API instability | MCP abstraction; graceful degradation; retries |
| Cost blowup from many model calls | Per-run budgets; cheaper models for narrow subagents; cost in the audit log |
| Memory poisoning / stale conventions | User-editable memory; provenance on stored decisions |
| Research participant data mishandling | Consent-scoped storage; PII minimization; approval gate on any participant contact |

---

## 14. Open questions
- Which model(s) for orchestrator vs. subagents — one model, or a cheaper model for narrow roles?
- Build the orchestration loop in-house or on an existing agent framework?
- How much memory is auto-written vs. explicitly curated by the user?
- Figma integration: live connector, or export-based, for v1?
- Do we need a persistent participant-recruiting connector in v1, or manual for now?

---

## 15. Out of scope (v1)
Multi-tenant SaaS; no-code workflow builder; non-dev/UX domains; autonomous irreversible actions; custom model training.

---

## Appendix — glossary
- **Orchestrator:** the single generalist agent that owns the goal, plans, and delegates.
- **Subagent:** a specialist worker with scoped prompt, tools, and isolated context.
- **HITL gate:** a human-in-the-loop approval checkpoint for risky actions.
- **Domain graph / memory moat:** accumulated proprietary context (codebase conventions, research repository) that makes the system sharper over time.
- **MCP:** Model Context Protocol — the standard used to connect external tools/data.
