"""Record what a tool was called with and what it returned.

One decorator, applied inside `@app.function` so Modal still sees the real
function:

    @gpu_function
    @records("proteinmpnn/design_sequences")
    def design_sequences(structure_uri: str, ...) -> dict: ...

It writes a row when the call starts and completes it when the call ends, so a
container killed mid-run leaves a `running` row rather than no trace at all.
Results go in whole as jsonb: tool returns differ in shape and already carry
their artifact URIs, so nothing needs a schema per tool.

**Without `DATABASE_URL` set, this does nothing.** Recording is meant to be
addable to a tool without making the tool depend on a database being up, and
every tool here worked before this existed. When the URL *is* set but a write
fails, the failure is printed and the call proceeds: losing a row is bad, and
losing ten GPU-minutes because a row could not be written is worse.
"""

from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

URL_ENV = "DATABASE_URL"


@dataclass(frozen=True)
class Runtime:
    """What a tool reaches outside itself.

    A tool takes this as its first argument instead of importing a storage
    client, minting its own run id and making its own scratch directory. All of
    it is still the same code underneath; what changes is that a tool no longer
    names any of it, so replacing one is a change here rather than in every
    tool.

    Attributes:
        run_id: This call's identifier, and the folder its artifacts go in.
        name: What the scratch directory is named after.
        work: A scratch directory of this call's own, already created.
        storage: `upload`, `exists`, `parse_uri`, `run_prefix`. Inputs come in
            through `fetch` rather than `storage.download`.
        db: This module. Recording does nothing without `DATABASE_URL`, which
            is what lets a tool run with no database up.
        cache: The weights cache this tool was declared with, or None. Opaque:
            a tool touches it only through `save_cache`.
    """

    run_id: str
    name: str = "tool"
    work: Path = field(default=None)  # type: ignore[assignment]
    storage: ModuleType = field(default=None)  # type: ignore[assignment]
    db: ModuleType = field(default=None)  # type: ignore[assignment]
    cache: Any = None

    def __post_init__(self) -> None:
        # Filled in here rather than as class defaults so that `storage`, the
        # heavier import, is only paid for when a runtime is actually built.
        import sys

        from tools.common import shell
        from tools.common import storage as storage_module

        object.__setattr__(self, "work", self.work or shell.workspace(self.name))
        object.__setattr__(self, "storage", self.storage or storage_module)
        object.__setattr__(self, "db", self.db or sys.modules[__name__])

    @classmethod
    def create(cls, name: str = "tool", cache: Any = None) -> "Runtime":
        """A runtime for one call, with a fresh run id."""
        from tools.common import storage

        return cls(run_id=storage.new_run_id(), name=name, cache=cache)

    def prefix(self, *parts: str):
        """Where this call writes, optionally joined with a file name."""
        return self.storage.run_prefix(self.run_id).join(*parts)

    def fetch(self, uri: str, stem: str) -> Path:
        """Download an input into this call's scratch directory.

        The stored extension is kept, because the structure readers downstream
        choose their parser by it, and defaults to `.pdb` when the URI has
        none. Every tool that reads a structure was writing these three lines
        itself.

        Args:
            uri: Where the file is.
            stem: What to call it locally, without an extension.
        """
        suffix = Path(self.storage.parse_uri(uri).key).suffix.lower() or ".pdb"
        return self.storage.download(uri, self.work / f"{stem}{suffix}")

    def save_cache(self) -> None:
        """Publish whatever was just written to the weights cache.

        Called right after a model is loaded. Modal persists a volume when the
        call returns anyway, so this only moves that forward: a checkpoint
        downloaded by a 30-minute call becomes visible to the next container
        immediately rather than half an hour later. Does nothing when the tool
        was declared without a cache.
        """
        if self.cache is not None:
            self.cache.commit()


def _connect():
    url = os.environ.get(URL_ENV, "").strip()
    if not url:
        return None
    import psycopg

    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=10)
    except Exception as exc:  # noqa: BLE001 - never fail a call over a record
        print(f"[records] no database, not recording: {exc}", flush=True)
        return None


def _json(value: Any) -> str:
    """Serialise for jsonb, replacing anything unserialisable with its repr.

    A tool may return a Path or a numpy scalar in some corner, and that must
    not be what fails a run that otherwise succeeded.
    """
    return json.dumps(value, default=repr)


def start(tool: str, run_id: str, params: dict) -> int | None:
    """Record that a call began, returning the row id to complete later."""
    connection = _connect()
    if connection is None:
        return None
    try:
        with connection:
            row = connection.execute(
                "insert into tool_runs (run_id, tool, status, params) "
                "values (%s, %s, 'running', %s) returning id",
                (run_id, tool, _json(params)),
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:  # noqa: BLE001 - never fail a call over a record
        print(f"[records] could not open a row for {tool}: {exc}", flush=True)
        return None
    finally:
        connection.close()


def finish(
    row_id: int | None,
    status: str,
    result: Any = None,
    error: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Complete a previously opened row."""
    if row_id is None:
        return
    connection = _connect()
    if connection is None:
        return
    try:
        with connection:
            connection.execute(
                "update tool_runs set status = %s, result = %s, error = %s, "
                "finished_at = now(), duration_ms = %s where id = %s",
                (status, _json(result) if result is not None else None,
                 error, duration_ms, row_id),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[records] could not close row {row_id}: {exc}", flush=True)
    finally:
        connection.close()


def records(tool: str) -> Callable:
    """Record each call of the wrapped tool.

    Args:
        tool: How the tool is addressed, `"<modal app>/<function>"`, matching
            how a workflow names it.

    The run id is taken from the returned dict, since every tool here mints
    one and returns it, and it doubles as the artifact prefix in S3. A tool
    that returns no `run_id` is still recorded, with an empty one.
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # The row opens before the run id exists, since the point is to
            # leave a trace even if the call dies. It is filled in on finish.
            row_id = start(tool, "", kwargs if kwargs else {"args": list(args)})
            began = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                finish(
                    row_id, "failed", error=f"{type(exc).__name__}: {exc}"[:4000],
                    duration_ms=int((time.monotonic() - began) * 1000),
                )
                raise

            run_id = result.get("run_id", "") if isinstance(result, dict) else ""
            if row_id is not None and run_id:
                _set_run_id(row_id, run_id)
            finish(
                row_id, "succeeded", result=result,
                duration_ms=int((time.monotonic() - began) * 1000),
            )
            return result

        return wrapper

    return decorate


def start_workflow(workflow: str, document: str, inputs: dict) -> int | None:
    """Record that a workflow run began, returning its row id.

    The document is stored whole, not just its name, so a run can be read back
    after the file on disk has changed.
    """
    connection = _connect()
    if connection is None:
        return None
    try:
        with connection:
            row = connection.execute(
                "insert into workflow_runs "
                "(workflow, document, inputs, status) "
                "values (%s, %s, %s, 'running') returning id",
                (workflow, document, _json(inputs)),
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:  # noqa: BLE001 - never fail a run over a record
        print(f"[records] could not open a workflow row: {exc}", flush=True)
        return None
    finally:
        connection.close()


def finish_workflow(
    row_id: int | None,
    status: str,
    steps: Any = None,
    output: Any = None,
    error: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Complete a workflow row with its per-step report and reduced output."""
    if row_id is None:
        return
    connection = _connect()
    if connection is None:
        return
    try:
        with connection:
            connection.execute(
                "update workflow_runs set status = %s, steps = %s, "
                "output = %s, error = %s, finished_at = now(), "
                "duration_ms = %s where id = %s",
                (status, _json(steps if steps is not None else []),
                 _json(output) if output is not None else None,
                 error, duration_ms, row_id),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[records] could not close workflow row {row_id}: {exc}",
              flush=True)
    finally:
        connection.close()


def link_calls(
    workflow_run_id: int | None, step: str, run_ids: list[str]
) -> int:
    """Attach already-recorded tool calls to the workflow step that made them.

    Linking happens after the fact rather than being passed into each tool,
    because a tool executes in its own container and does not inherit the
    runner's environment. Since every tool returns the run id it minted, the
    runner can match on that and no tool signature grows workflow plumbing it
    would only pass through.

    Returns:
        How many rows were linked, which is worth comparing against the number
        of calls: a shortfall means a tool ran without recording, usually
        because its image predates the decorator.
    """
    if workflow_run_id is None or not run_ids:
        return 0
    connection = _connect()
    if connection is None:
        return 0
    try:
        with connection:
            result = connection.execute(
                "update tool_runs set workflow_run_id = %s, step = %s "
                "where run_id = any(%s) and workflow_run_id is null",
                (workflow_run_id, step, list(run_ids)),
            )
        return result.rowcount
    except Exception as exc:  # noqa: BLE001
        print(f"[records] could not link {step}: {exc}", flush=True)
        return 0
    finally:
        connection.close()


def _set_run_id(row_id: int, run_id: str) -> None:
    """Fill in the run id once the call has produced one."""
    connection = _connect()
    if connection is None:
        return
    try:
        with connection:
            connection.execute(
                "update tool_runs set run_id = %s where id = %s",
                (run_id, row_id),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[records] could not set run_id on {row_id}: {exc}", flush=True)
    finally:
        connection.close()
