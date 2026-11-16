"""Prepare demonstration pairs and a local or subprocess evaluation session."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from .config import RunConfig


def prepare_policy(config, samples, manifest, *, pairs_only=False, overwrite=False):
    from .config.base import fingerprint
    from .data import build_pair_set, load_pairs
    from .objective import build_objective
    from .policies import ContextSource, LatentBank, build_policy
    from .run.bank import EndpointBank
    from .search.pso import ObjectiveEvaluator

    if not pairs_only and config.solver.context_cache < len(samples):
        raise ValueError("context_cache must hold every calibration observation")
    policy = build_policy(config.policy, config.data)
    metadata = policy.metadata()
    contexts = None
    if not pairs_only:
        contexts = ContextSource(policy, samples, cache_size=config.solver.context_cache)
    pair_manifest = build_pair_set(
        policy,
        samples,
        config.pairs_dir,
        data_fingerprint=config.data_fingerprint,
        dataset_manifest=manifest,
        overwrite=overwrite,
        get_context=contexts.get if contexts is not None else None,
        progress=(lambda line: print(line, file=sys.stderr, flush=True)) if pairs_only else None,
    )
    info = {"metadata": metadata, "num_pairs": pair_manifest["num_pairs"]}
    if pairs_only:
        return info, None

    pairs = load_pairs(config.pairs_dir)
    latents = LatentBank(
        policy, noise_offset=config.method.noise_offset, n_latents=config.method.n_latents
    )
    objective = build_objective(
        policy,
        pairs,
        contexts,
        latents,
        config.method,
        budget=config.budget,
        micro_batch=config.solver.micro_batch,
        score_horizon=int(getattr(policy, "score_horizon", 0) or 0),
        score_slice=getattr(policy, "score_slice", None),
        scratch=config.endpoints_dir,
    )
    identity = fingerprint(
        {
            "schema": "gys-rollout",
            "pairs": pair_manifest["sha256"],
            "config": config.to_dict(),
            "policy": metadata,
        }
    )
    bank = EndpointBank(config.endpoints_dir, identity=identity)
    return info, ObjectiveEvaluator(objective, bank)


class Session:
    """Expose policy metadata and the evaluator for a single run."""

    def __init__(self, info, evaluator):
        self.metadata = info["metadata"]
        self.num_pairs = info["num_pairs"]
        self.evaluator = evaluator

    def calibrate(self, seeds, grid):
        return self.evaluator.calibrate(seeds, grid)

    def score(self, nodes, order):
        return self.evaluator.score(nodes, order)


@contextmanager
def evaluator_session(
    config: RunConfig,
    policy_python: Path | None,
    *,
    pairs_only: bool = False,
    overwrite: bool = False,
):
    from .data import build_dataset

    dataset = build_dataset(config.data)
    manifest = dataset.manifest()
    if policy_python is None:
        info, evaluator = prepare_policy(
            config, list(dataset), manifest, pairs_only=pairs_only, overwrite=overwrite
        )
        yield Session(info, evaluator)
    else:
        from .run.policy_process import PolicyProcess

        with PolicyProcess.start(policy_python) as child:
            info = child.prepare(config, dataset, manifest, pairs_only, overwrite)
            yield Session(info, child)
