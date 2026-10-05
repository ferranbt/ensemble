-- One row per tool invocation.
--
-- Deliberately one table with jsonb rather than a table per tool. Tool results
-- are already plain dicts of differing shapes, and a schema per tool would
-- have to change every time a tool's return does. Filtering and projection are
-- ordinary operations over jsonb, so nothing is lost by storing it whole.

create table if not exists tool_runs (
    id           bigserial primary key,

    -- The run id the tool minted, which is also its S3 prefix, so a row leads
    -- to its artifacts and back.
    run_id       text not null,

    -- "<modal app>/<function>", matching how a workflow addresses a tool.
    tool         text not null,

    status       text not null check (status in ('running', 'succeeded', 'failed')),

    -- What it was called with, and what came back. `result` is null while a
    -- call is still running or if it failed.
    params       jsonb not null default '{}'::jsonb,
    result       jsonb,
    error        text,

    started_at   timestamptz not null default now(),
    finished_at  timestamptz,
    duration_ms  integer
);

create index if not exists tool_runs_run_id_idx on tool_runs (run_id);
create index if not exists tool_runs_tool_idx   on tool_runs (tool, started_at desc);

-- A call that never reported back. `finished_at` is null both while running and
-- when a container was killed outright, so treat anything older than the
-- tool's timeout as lost rather than in flight.
create index if not exists tool_runs_running_idx
    on tool_runs (started_at) where status = 'running';
