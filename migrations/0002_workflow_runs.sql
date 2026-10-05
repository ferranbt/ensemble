-- One row per workflow execution, and a link from each tool call to the step
-- that made it.
--
-- Separate from `tool_runs` rather than another row in it, for two reasons. A
-- workflow run carries different things: the document it ran, the inputs, and
-- a per-step report. And a tool call must be able to exist without a workflow,
-- since calling a tool directly is still a call worth recording. Folding both
-- into one table would leave half its columns null for each kind.

create table if not exists workflow_runs (
    id           bigserial primary key,

    -- The document's own name, and the document itself, so a run can be
    -- reproduced or audited after the file has changed.
    workflow     text not null,
    document     text not null,

    inputs       jsonb not null default '{}'::jsonb,

    status       text not null check (status in ('running', 'succeeded', 'failed')),

    -- Per-step arity, successes and failures. Kept here rather than derived
    -- from tool_runs because a step that was filtered down to zero calls made
    -- no tool calls at all, and that is a fact worth keeping.
    steps        jsonb not null default '[]'::jsonb,

    -- The reduced result the document's `output` block asked for.
    output       jsonb,
    error        text,

    started_at   timestamptz not null default now(),
    finished_at  timestamptz,
    duration_ms  integer
);

create index if not exists workflow_runs_name_idx
    on workflow_runs (workflow, started_at desc);

-- Tie each tool call to the workflow step that made it. Nullable throughout:
-- a direct call has no workflow, and that is not an error.
alter table tool_runs
    add column if not exists workflow_run_id bigint references workflow_runs (id),
    add column if not exists step text;

create index if not exists tool_runs_workflow_idx
    on tool_runs (workflow_run_id, step);
