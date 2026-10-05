from __future__ import annotations

import pytest

from tools.common.shell import ToolError, batches_for, run


class TestBatchesFor:
    @pytest.mark.parametrize(
        "total, requested, expected",
        [
            (8, 8, 8),      # exact fit
            (8, 16, 8),     # request larger than total
            (10, 8, 5),     # 8 does not divide 10, step down to 5
            (7, 8, 7),      # prime total, one batch
            (9, 4, 3),      # step down to the nearest divisor
            (1, 8, 1),      # single item
            (100, 8, 5),    # 8 does not divide 100; 5 does
        ],
    )
    def test_returns_a_divisor_no_larger_than_requested(
        self, total, requested, expected
    ):
        assert batches_for(total, requested) == expected

    @pytest.mark.parametrize("total", range(1, 40))
    @pytest.mark.parametrize("requested", [1, 2, 4, 8, 16])
    def test_result_always_divides_the_total(self, total, requested):
        """ProteinMPNN drops samples when batch size does not divide the count.

        It computes num_batches = total // batch_size, so any remainder is
        silently lost. The chosen size must divide exactly, every time.
        """
        size = batches_for(total, requested)
        assert total % size == 0
        assert 1 <= size <= min(requested, total)

    def test_rejects_a_nonsensical_total(self):
        with pytest.raises(ValueError):
            batches_for(0, 8)


class TestRun:
    def test_returns_captured_output(self):
        assert "hello" in run(["echo", "hello"])

    def test_merges_stderr_into_the_result(self):
        assert "oops" in run(["sh", "-c", "echo oops >&2"])

    def test_raises_with_the_command_and_its_output(self):
        with pytest.raises(ToolError) as excinfo:
            run(["sh", "-c", "echo boom >&2; exit 3"])
        assert excinfo.value.returncode == 3
        assert "boom" in excinfo.value.output
        assert "boom" in str(excinfo.value)

    def test_accepts_non_string_arguments(self):
        """Callers pass ints and Paths straight through from tool parameters."""
        assert "37" in run(["echo", 37])

    def test_runs_in_the_given_directory(self, tmp_path):
        (tmp_path / "marker.txt").write_text("here")
        assert "marker.txt" in run(["ls"], cwd=tmp_path)


class TestToolErrorSurvivesPickling:
    """A tool failure must cross a process boundary as itself.

    Modal rebuilds a remote exception locally by calling the class with its
    `args`. RuntimeError sets `args` to the single formatted message, which
    this constructor cannot accept, so the deserialisation failure gets
    reported instead of the actual failure. That is how a missing input file
    once surfaced as a complaint about missing arguments.
    """

    def error(self):
        from tools.common.shell import ToolError

        return ToolError(["python", "run_inference.py"], 1, "boom")

    def test_round_trips(self):
        import pickle

        restored = pickle.loads(pickle.dumps(self.error()))
        assert restored.cmd == ["python", "run_inference.py"]
        assert restored.returncode == 1
        assert restored.output == "boom"

    def test_the_message_survives(self):
        import pickle

        original = self.error()
        assert str(pickle.loads(pickle.dumps(original))) == str(original)

    def test_names_the_script_that_failed(self):
        assert "run_inference.py exited 1" in str(self.error())
