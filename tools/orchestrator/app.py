"""Run a workflow document, as a Modal action.

One action, `run_workflow`: parse a document, run each step, and pass results
along. A step fans out over a prior step's results or a workflow input, and
each element becomes its own call.

**The runner imports no tool.** It resolves each by name at call time:

    modal.Function.from_name("msa", "search_msa")

so it shares no image with any tool, needs no GPU, and a tool added later is
usable in a workflow with no change here.

**Results are held in memory for now.** That is fine for the tens of items this
is built for and is the first thing to replace: a real campaign needs its
results in Postgres so a filter is a query and a caller gets a summary rather
than everything.

    modal run tools/orchestrator/app.py \\
        --workflow workflows/msa_then_fold.yaml \\
        --inputs '{"targets": ["MKTAYIAK...", "GWSTELEK..."]}'
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import modal

from tools.common import records
from tools.common.images import SECRETS, base_image, with_local_sources
from tools.orchestrator import document

# No GPU. This waits on other functions, which may take hours between them.
TIMEOUT = 12 * 60 * 60

app = modal.App("orchestrator")

image = with_local_sources(base_image().pip_install("pyyaml"))


@app.function(image=image, timeout=TIMEOUT, secrets=SECRETS)
def run_workflow(
    workflow: str,
    inputs: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a workflow document and return its reduced output.

    Args:
        workflow: The document itself, as YAML.
        inputs: Values for the workflow's declared inputs.
        dry_run: Resolve and report each step's arity without calling
            anything. The cheap way to check a document before spending GPU
            time, since a fan-out is easy to write and expensive to get wrong.

    Returns:
        `output`, projected as the document's `output` block says, plus
        `steps` recording each step's arity, failures and the tool that ran.
    """
    began = time.monotonic()
    parsed = document.parse(workflow)
    bound = document.bind_inputs(parsed, inputs or {})

    # A dry run makes no calls and produces no artifacts, so it is not worth a
    # row. Everything else is recorded before the first step, so a run killed
    # part way leaves a `running` row rather than no trace.
    workflow_row = (
        None if dry_run
        else records.start_workflow(parsed.name, workflow, bound)
    )

    scope: dict[str, Any] = {"inputs": bound}
    results: dict[str, list] = {}
    report: list[dict[str, Any]] = []

    try:
        for step in parsed.steps:
            calls = document.plan(step, scope)
            print(f"[{step.id}] {step.tool}: {len(calls)} call(s)", flush=True)

            if dry_run:
                # A step that emits produces an unknown number of items per
                # call, so every arity after it is a floor, not a count. Said
                # plainly, because a dry run that quietly understates a fan-out
                # is worse than no dry run at all.
                report.append({
                    "step": step.id,
                    "tool": step.tool,
                    "calls": len(calls),
                    "calls_are_lower_bound": _downstream_of_emitter(parsed, step),
                })
                # Stand in one result per call so arity propagates through the
                # rest of the document, which is what makes a dry run useful.
                placeholders = [document.Unknown()] * len(calls)
                scope[step.id] = {
                    "results": placeholders,
                    "items": document.emitted(step, placeholders),
                }
                continue

            outcomes, failures = _run_step(step, calls)
            results[step.id] = outcomes
            scope[step.id] = {
                "results": outcomes,
                "items": document.emitted(step, outcomes),
            }

            linked = records.link_calls(
                workflow_row, step.id,
                [o["run_id"] for o in outcomes
                 if isinstance(o, dict) and o.get("run_id")],
            )
            report.append({
                "step": step.id,
                "tool": step.tool,
                "calls": len(calls),
                "succeeded": len(outcomes),
                "failed": len(failures),
                "linked": linked,
                "errors": failures[:5],
            })

            if not outcomes:
                raise RuntimeError(
                    f"Step {step.id} produced no results from {len(calls)} "
                    f"call(s), so nothing downstream can run. First error: "
                    f"{failures[0] if failures else 'none reported'}"
                )

        output = None if dry_run else document.reduce_output(parsed, results)
    except BaseException as exc:
        records.finish_workflow(
            workflow_row, "failed", steps=report,
            error=f"{type(exc).__name__}: {exc}"[:4000],
            duration_ms=int((time.monotonic() - began) * 1000),
        )
        raise

    records.finish_workflow(
        workflow_row, "succeeded", steps=report, output=output,
        duration_ms=int((time.monotonic() - began) * 1000),
    )

    return {
        "workflow": parsed.name,
        "workflow_run_id": workflow_row,
        "dry_run": dry_run,
        "steps": report,
        "output": output,
    }


def _downstream_of_emitter(workflow: document.Workflow, step: document.Step) -> bool:
    """Whether this step's arity depends on how many items an earlier step emits."""
    for earlier in workflow.steps:
        if earlier.id == step.id:
            return False
        if earlier.emits:
            return True
    return False


def _run_step(
    step: document.Step, calls: list[tuple[dict, dict]]
) -> tuple[list, list[str]]:
    """Invoke one step's calls in parallel, keeping failures separate.

    A failed call must not look like a filtered-out one, so errors are
    collected and reported rather than silently shrinking the result set.

    Each result is attached the lineage of the item its call was made from,
    under `carried`, nested so it cannot overwrite a field the tool returned.
    """
    fn = modal.Function.from_name(*step.tool.split("/"))
    pending = [(fn.spawn(**params), carried) for params, carried in calls]

    outcomes: list[Any] = []
    failures: list[str] = []
    for index, (call, carried) in enumerate(pending):
        try:
            outcome = call.get()
        except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
            failures.append(f"call {index}: {type(exc).__name__}: {exc}")
            print(f"[{step.id}] call {index} failed: {exc}", flush=True)
            continue
        if carried and isinstance(outcome, dict):
            outcome = {**outcome, document.CARRIED: carried}
        outcomes.append(outcome)
    return outcomes, failures


@app.local_entrypoint()
def main(
    workflow: str,
    inputs: str = "{}",
    dry_run: bool = False,
    out: str = "",
):
    """Run a workflow from the command line.

    Args:
        workflow: Path to a YAML workflow document.
        inputs: Input values as JSON, e.g. '{"targets": ["MKT..."]}'.
        dry_run: Report each step's arity without running anything.
        out: Optional path to write the full result as JSON.
    """
    import json

    result = run_workflow.remote(
        workflow=Path(workflow).read_text(),
        inputs=json.loads(inputs),
        dry_run=dry_run,
    )

    print(f"\nworkflow {result['workflow']}"
          f"{' (dry run)' if result['dry_run'] else ''}")
    for step in result["steps"]:
        counts = f"{step['calls']} call(s)"
        if step.get("calls_are_lower_bound"):
            # An earlier step emits an unknown number of items per call, so
            # this figure is a floor. Saying so matters: the point of a dry run
            # is catching a fan-out that is larger than intended.
            counts += " or more"
        if not result["dry_run"]:
            counts += f", {step['succeeded']} ok, {step['failed']} failed"
        print(f"  {step['step']:<16} {step['tool']:<34} {counts}")
        for error in step.get("errors", []):
            print(f"    {error}")

    if result["output"] is not None:
        print(f"\noutput: {json.dumps(result['output'], indent=2)[:2000]}")

    if out:
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {out}")
