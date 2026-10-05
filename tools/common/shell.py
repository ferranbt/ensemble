"""Subprocess helpers shared by every wrapped command-line tool."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class ToolError(RuntimeError):
    """A wrapped command-line tool exited non-zero."""

    def __init__(self, cmd: list[str], returncode: int, output: str):
        self.cmd = cmd
        self.returncode = returncode
        self.output = output
        super().__init__(
            f"{Path(cmd[1] if len(cmd) > 1 else cmd[0]).name} exited {returncode}\n"
            f"$ {' '.join(cmd)}\n{output.strip()[-4000:]}"
        )

    def __reduce__(self):
        """How to rebuild this when it crosses a process boundary.

        An exception is unpickled by calling the class with `args`, and
        `RuntimeError` sets `args` to the single formatted message, which this
        constructor cannot accept. Modal then fails to deserialize the failure
        and reports that instead of the failure itself, which is how a missing
        input file once surfaced as a complaint about missing arguments.
        """
        return (self.__class__, (self.cmd, self.returncode, self.output))


def run(cmd: list[str], cwd: str | Path | None = None) -> str:
    """Run a command, stream nothing, capture everything, raise on failure.

    stdout and stderr are merged and echoed into the caller's logs so a Modal
    run shows the underlying tool's output, and returned so callers can parse
    values the tool only prints (ProteinMPNN reports scores this way).
    """
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise ToolError(cmd, proc.returncode, proc.stdout or "")
    return proc.stdout or ""


def workspace(prefix: str) -> Path:
    """Create a scratch directory inside the container for one invocation."""
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_"))


def batches_for(total: int, batch_size: int) -> int:
    """Largest batch size <= `batch_size` that divides `total` exactly.

    ProteinMPNN computes `num_batches = num_seq_per_target // batch_size` and
    silently produces fewer results than requested when the division has a
    remainder, so callers must not pass a batch size that does not divide the
    requested count.
    """
    if total < 1:
        raise ValueError("total must be at least 1")
    size = max(1, min(batch_size, total))
    while total % size:
        size -= 1
    return size
