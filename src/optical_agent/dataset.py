"""Generate related prescription families, split before fitting any model."""

import json
from pathlib import Path

import numpy as np

from .physics import evaluate


def generate(path, families=250, members=40, seed=2026):
    if families < 30 or members < 5:
        raise ValueError("at least 30 families and 5 members required")
    rng = np.random.default_rng(seed)
    family_tasks = np.column_stack(
        [
            rng.uniform(25, 80, families),
            rng.uniform(2, 10, families),
            rng.uniform(3.5, 9, families),
        ]
    )
    anchors = rng.uniform(0.1, 0.9, (families, 9))
    anchors_glass = rng.integers(0, 3, (families, 2))
    ids = np.repeat(np.arange(families), members)
    genes = np.clip(anchors[ids] + rng.normal(0, 0.23, (families * members, 9)), 0, 1)
    glasses = anchors_glass[ids].copy()
    mutate = rng.random(glasses.shape) < 0.25
    glasses[mutate] = rng.integers(0, 3, mutate.sum())
    tasks = family_tasks[ids]
    order = rng.permutation(families)
    split_by_family = np.full(families, 2, dtype=np.int64)
    split_by_family[order[: int(0.8 * families)]] = 0
    split_by_family[order[int(0.8 * families) : int(0.9 * families)]] = 1
    metrics = {}
    for start in range(0, len(genes), 128):
        out = evaluate(
            genes[start : start + 128],
            glasses[start : start + 128],
            tasks[start : start + 128],
        )
        for key, val in out.items():
            metrics.setdefault(key, []).append(val)
    metrics = {k: np.concatenate(v) for k, v in metrics.items()}
    # Every member is simulated once, including infeasible examples.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        genes=genes,
        glasses=glasses,
        tasks=tasks,
        family_id=ids,
        split=split_by_family[ids],
        family_tasks=family_tasks,
        **metrics,
    )
    manifest = {
        "seed": seed,
        "designs": len(genes),
        "families": families,
        "members_per_family": members,
        "split_names": {"0": "train", "1": "validation", "2": "test"},
        "family_ids": {
            name: np.flatnonzero(split_by_family == i).tolist()
            for i, name in enumerate(["train", "validation", "test"])
        },
        "simulator_calls": len(genes),
        "rings": 3,
        "spokes": 8,
        "feasible_designs": int(metrics["feasible"].sum()),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def elite_indices(data, split, fraction=0.2):
    """Per-family objective ranking uses only the requested split."""
    indices = []
    for family in np.unique(data["family_id"][data["split"] == split]):
        candidates = np.flatnonzero(
            (data["family_id"] == family) & (data["split"] == split)
        )
        count = max(1, int(len(candidates) * fraction))
        indices.extend(candidates[np.argsort(data["objective"][candidates])[:count]])
    return np.asarray(indices)
