"""Exercise the pipe protocol and search through a separate interpreter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gys.config import load_config
from gys.data.records import CalibRecord, CalibSample
from gys.run import policy_process
from gys.search.pso import search

_FAKE_POLICY = """
import numpy as np
from gys import runtime

class Evaluator:
    def calibrate(self, seeds, grid):
        order = float(grid[0])
        return [order] * len(seeds), order, 1.0, [self.score(x, order) for x in seeds]

    def score(self, nodes, order):
        if order == 99:
            raise ValueError("invalid order")
        return float(np.square(nodes[1:-1] - 0.5).sum())

def prepare(config, samples, manifest, **options):
    assert config.task == "test" and config.budget == 4
    sample, = samples
    assert sample.record.noise_seed == 123
    assert sample.observation["prompt"] == "move the cube"
    assert sample.observation["head"].dtype == np.uint8
    assert sample.observation["head"].tolist() == [[[255, 33, 0]]]
    assert sample.gt_action_raw.shape == (2, 14)
    assert manifest == {"example": True}
    os.write(1, b"policy log from native stdout\\n")
    return {"metadata": {"wire_knot_axis": "canonical"}, "num_pairs": 1}, Evaluator()

runtime.prepare_policy = prepare
"""


def test_subprocess_search_and_error_reporting(monkeypatch, tmp_path):
    monkeypatch.setattr(
        policy_process,
        "_BOOTSTRAP",
        policy_process._BOOTSTRAP.replace("with reply:", _FAKE_POLICY + "\nwith reply:"),
    )
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/lerobot_pi05.yaml",
        task="test",
        budget=4,
        run_root=tmp_path,
    )
    sample = CalibSample(
        CalibRecord("sample", 0, 0, 0, 2, 123),
        {"head": np.array([[[255, 33, 0]]], dtype=np.uint8), "prompt": "move the cube"},
        np.zeros((2, 14), dtype=np.float32),
    )
    with policy_process.PolicyProcess.start(Path(sys.executable)) as child:
        info = child.prepare(config, [sample], {"example": True}, False, False)
        assert info == {"metadata": {"wire_knot_axis": "canonical"}, "num_pairs": 1}
        solution = search(child, budget=4, method=config.method, solver=config.solver)
        assert solution.evaluations == config.solver.evaluation_budget()
        assert solution.loss == min(point.loss for point in solution.measured)
        with pytest.raises(RuntimeError, match="policy process: ValueError: invalid order"):
            child.score(np.array([0.0, 0.5, 1.0]), 99)
    assert child.process.poll() == 0
