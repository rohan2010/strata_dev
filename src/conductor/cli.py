"""Conductor CLI.

Only the inspection commands exist so far — `run` arrives with the LangGraph skeleton
(EXECUTION-PLAN.md tasks 0.4/0.7). What is here answers "which model serves which job,
and does the config actually hang together", which is the question you cannot answer by
reading four YAML files at once.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from conductor.audit import AuditStore
from conductor.connectors import ConnectorRegistry, GitHubConnector
from conductor.models import Capability, ModelClient, ModelRegistry, clamp_effort
from conductor.orchestration import Orchestrator, sqlite_checkpointer
from conductor.subagents import SubagentRegistry

app = typer.Typer(help="Multi-agent orchestrator", no_args_is_help=True)
models_app = typer.Typer(help="Inspect model routing", no_args_is_help=False)
app.add_typer(models_app, name="models")

console = Console()


def _load_models(path: Path) -> ModelRegistry:
    try:
        return ModelRegistry.from_file(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]model registry:[/red] {exc}")
        raise typer.Exit(1) from exc


@models_app.callback(invoke_without_command=True)
def models(
    ctx: typer.Context,
    registry_path: Path = typer.Option(Path("models.yaml"), "--registry", "-r"),
) -> None:
    """Show which model serves which role, and what each one costs."""
    if ctx.invoked_subcommand is not None:
        return

    registry = _load_models(registry_path)
    subagents_by_route: dict[str, list[str]] = {}
    try:
        for m in SubagentRegistry.from_dir().all():
            subagents_by_route.setdefault(m.route, []).append(m.name)
    except (FileNotFoundError, ValueError):
        pass  # routes are inspectable with no subagents defined yet

    routes = Table(title="Routes", header_style="bold")
    routes.add_column("role")
    routes.add_column("preferred")
    routes.add_column("fallbacks")
    routes.add_column("used by")

    for role in sorted(registry.routes):
        chain = [registry.models[a] for a in registry.routes[role]]
        preferred = f"{chain[0].alias} [dim]({chain[0].provider})[/dim]"
        fallbacks = " -> ".join(f"{m.alias}" for m in chain[1:]) or "[dim]none[/dim]"
        label = f"{role} [dim](default)[/dim]" if role == registry.default_route else role
        routes.add_row(label, preferred, fallbacks, ", ".join(subagents_by_route.get(role, [])))
    console.print(routes)

    models_table = Table(title="Models", header_style="bold")
    for col in ("alias", "provider", "id", "$/MTok in-out", "capabilities"):
        models_table.add_column(col)

    for alias in sorted(registry.models):
        spec = registry.models[alias]
        price = (
            f"{spec.price.input:g} / {spec.price.output:g}"
            if spec.price
            else "[yellow]unpriced[/yellow]"
        )
        caps = ", ".join(sorted(spec.supports)) or "[dim]none[/dim]"
        models_table.add_row(alias, str(spec.provider), spec.id, price, caps)
    console.print(models_table)

    if unpriced := registry.unpriced():
        console.print(
            f"[yellow]![/yellow] unpriced: {', '.join(unpriced)} "
            "— these run, but contribute tokens without dollars to cost reports"
        )


@models_app.command("effort")
def effort_matrix(
    registry_path: Path = typer.Option(Path("models.yaml"), "--registry", "-r"),
) -> None:
    """Show how each subagent's declared effort lands on its actual provider.

    Effort clamps downward per provider, so a manifest asking for `xhigh` gets `high` on
    OpenAI and Gemini. Worth seeing explicitly before wondering why a fallback run felt
    shallower than the primary.
    """
    registry = _load_models(registry_path)
    try:
        subagents = SubagentRegistry.from_dir()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]subagents:[/red] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Declared effort vs. delivered", header_style="bold")
    for col in ("subagent", "route", "declared", "model", "delivered"):
        table.add_column(col)

    for m in subagents.all():
        for spec in registry.chain(m.route):
            delivered = clamp_effort(spec.provider, m.effort)
            if delivered is None:
                shown = "[dim]n/a (on/off)[/dim]"
            elif delivered != m.effort:
                shown = f"[yellow]{delivered}[/yellow]"
            else:
                shown = str(delivered)
            table.add_row(m.name, m.route, str(m.effort), f"{spec.alias}", shown)
    console.print(table)


@app.command()
def run(
    goal: str = typer.Argument(..., help="What you want done, in plain language"),
    repo: str = typer.Option(..., "--repo", help="Target repository, 'owner/name'"),
    registry_path: Path = typer.Option(Path("models.yaml"), "--registry", "-r"),
    show_plan: bool = typer.Option(True, "--show-plan/--no-show-plan"),
) -> None:
    """Plan a goal, dispatch it to subagents, and report the outcome.

    Approvals are not gated yet — the policy engine is Phase 1 (task 1.1). Until then the
    Implementer's write scopes are live, which is why `--repo` is required and explicit
    rather than inferred from the working directory.
    """
    registry = _load_models(registry_path)
    audit = AuditStore()
    connectors = ConnectorRegistry()
    connectors.register(GitHubConnector(repo))

    try:
        subagents = SubagentRegistry.from_dir()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]subagents:[/red] {exc}")
        raise typer.Exit(1) from exc

    run_id = audit.start_run(goal)
    console.print(f"[dim]run {run_id} -> {repo}[/dim]")

    orchestrator = Orchestrator(
        client=ModelClient(registry, audit=audit, run_id=run_id),
        subagents=subagents,
        connectors=connectors,
        audit=audit,
        checkpointer=sqlite_checkpointer(),
    )
    final = orchestrator.run(goal, run_id=run_id)

    plan = final.get("plan")
    if show_plan and plan is not None:
        console.print(f"\n[bold]Plan[/bold]\n{plan.render()}\n")

    results = Table(title="Results", header_style="bold")
    for col in ("task", "subagent", "status", "tools", "model", "result"):
        results.add_column(col)
    for task_id, result in (final.get("results") or {}).items():
        colour = {"completed": "green", "failed": "red", "skipped": "yellow"}
        status = f"[{colour.get(result.status, 'white')}]{result.status}[/]"
        detail = result.error or result.summary
        results.add_row(
            task_id,
            result.subagent,
            status,
            str(result.tool_calls),
            result.model,
            (detail[:80] + "…") if len(detail) > 80 else detail,
        )
    console.print(results)

    total, unpriced = audit.run_cost_usd(run_id)
    cost = f"${total:.4f}" + (f" (+{unpriced} unpriced calls)" if unpriced else "")
    console.print(f"\n{final.get('summary', '')}  [dim]{cost}[/dim]")

    audit.close()
    if final.get("status") not in ("completed",):
        raise typer.Exit(1)


@app.command()
def doctor(
    registry_path: Path = typer.Option(Path("models.yaml"), "--registry", "-r"),
) -> None:
    """Boot-time checks: do the manifests, routes, and connectors agree?"""
    problems: list[str] = []
    registry = _load_models(registry_path)

    try:
        subagents = SubagentRegistry.from_dir()
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]subagents:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]ok[/green] {len(registry.models)} models, "
                  f"{len(registry.routes)} routes, {len(subagents)} subagents")

    for name, route in subagents.validate_routes(registry):
        problems.append(
            f"subagent {name!r} names route {route!r}, which models.yaml does not define"
        )

    for m in subagents.all():
        if registry.has_route(m.route):
            continue
        console.print(
            f"[yellow]![/yellow] {m.name}: no route {m.route!r}; "
            f"falling back to default_route {registry.default_route!r}"
        )

    # A subagent with tools needs a model that can call them.
    for m in subagents.all():
        if not m.tools:
            continue
        for spec in registry.chain(m.route):
            if not spec.can(Capability.TOOLS):
                problems.append(
                    f"{m.name} declares {len(m.tools)} tools but its route reaches "
                    f"{spec.alias!r}, which does not declare the `tools` capability"
                )

    connectors = ConnectorRegistry()
    connectors.register(GitHubConnector("owner/repo", token="unused-for-validation"))
    for m in subagents.all():
        if missing := connectors.validate_manifest(m):
            problems.append(f"{m.name} declares unknown tools: {', '.join(missing)}")

    if problems:
        for p in problems:
            console.print(f"[red]x[/red] {p}")
        raise typer.Exit(1)
    console.print("[green]ok[/green] manifests, routes, and connector scopes all agree")


if __name__ == "__main__":
    app()
