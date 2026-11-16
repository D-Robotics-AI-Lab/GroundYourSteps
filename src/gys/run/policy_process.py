"""A local policy subprocess with a private request and response channel."""

from __future__ import annotations

import io
import json
import subprocess
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import DataConfig, MethodConfig, RunConfig, SolverConfig, policy_config_from_dict
from ..data.records import CalibRecord, CalibSample

# Select GYS from the caller's source tree, then resolve its dependencies in the child.
_BOOTSTRAP = """import os
import sys
reply = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.path.insert(0, sys.argv.pop(1))
import gys
sys.path.pop(0)
from gys.run.policy_process import serve
with reply:
    serve(sys.stdin.buffer, reply)
"""


def _send(pipe, value):
    pipe.write(json.dumps(value, allow_nan=False).encode("utf-8") + b"\n")
    pipe.flush()


def _read_sample(pipe):
    record, size = json.loads(pipe.readline())
    payload = pipe.read(size)
    if len(payload) != size:
        raise EOFError("interrupted calibration sample transfer")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        observation = {
            key[4:]: arrays[key] for key in arrays.files if key.startswith("obs_")
        }
        observation["prompt"] = str(observation["prompt"].item())
        return CalibSample(CalibRecord(**record), observation, arrays["action"])


class PolicyProcess:
    """Send calibration observations once, then exchange grids and scalar results."""

    def __init__(self, process: subprocess.Popen):
        self.process = process

    @classmethod
    @contextmanager
    def start(cls, python: Path):
        # Keep a venv's bin/python path. Resolving its symlink would bypass the venv.
        interpreter = python.expanduser().absolute()
        source = Path(__file__).resolve().parents[2]
        process = subprocess.Popen(
            [str(interpreter), "-I", "-B", "-u", "-c", _BOOTSTRAP, str(source)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            yield cls(process)
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            process.stdout.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "policy process closed the response channel; see its stderr for details"
            )
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def prepare(self, config, dataset, manifest, pairs_only, overwrite):
        pipe = self.process.stdin
        try:
            _send(pipe, [config.to_dict(), manifest, len(dataset), pairs_only, overwrite])
            for sample in dataset:
                with io.BytesIO() as buffer:
                    np.savez(
                        buffer,
                        action=sample.gt_action_raw,
                        **{f"obs_{key}": value for key, value in sample.observation.items()},
                    )
                    with buffer.getbuffer() as payload:
                        _send(pipe, [sample.record.to_dict(), payload.nbytes])
                        pipe.write(payload)
            pipe.flush()
        except BrokenPipeError:
            pass  # The child may have sent an error reply.
        return self._receive()

    def _request(self, command, *args):
        try:
            _send(self.process.stdin, [command, *args])
        except BrokenPipeError:
            pass  # The child may have sent an error reply.
        return self._receive()

    def calibrate(self, seeds, grid):
        return self._request("calibrate", [nodes.tolist() for nodes in seeds], grid.tolist())

    def score(self, nodes, order):
        return self._request("score", nodes.tolist(), order)


def serve(requests, replies):
    """Run all model operations in the policy Python interpreter."""
    from ..runtime import prepare_policy

    try:
        payload, manifest, count, pairs_only, overwrite = json.loads(requests.readline())
        config = RunConfig(
            name=payload["name"],
            run_root=Path(payload["run_root"]),
            method=MethodConfig.from_dict(payload["method"]),
            policy=policy_config_from_dict(payload["policy"]),
            data=DataConfig.from_dict(payload["data"]),
            solver=SolverConfig.from_dict(payload["solver"]),
            task=payload["task"],
            budget=payload["budget"],
        )
        samples = [_read_sample(requests) for _ in range(count)]
        info, evaluator = prepare_policy(
            config, samples, manifest, pairs_only=pairs_only, overwrite=overwrite
        )
        _send(replies, {"result": info})
        if evaluator is None:
            return

        for line in requests:
            command, value, setting = json.loads(line)
            if command == "calibrate":
                result = evaluator.calibrate(np.asarray(value), np.asarray(setting))
            elif command == "score":
                result = evaluator.score(np.asarray(value), float(setting))
            else:
                raise ValueError(f"unknown policy request {command!r}")
            _send(replies, {"result": result})
    except BrokenPipeError:
        return
    except Exception as exc:
        traceback.print_exc()
        _send(replies, {"error": f"policy process: {type(exc).__name__}: {exc}"})
