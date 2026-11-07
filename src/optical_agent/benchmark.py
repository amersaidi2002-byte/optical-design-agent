"""Paired benchmark on unseen families; report every task and random seed."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .model import load_model
from .physics import evaluate
from .search import optimize


def summarize(rows):
    baseline = np.array([r["baseline"]["metrics"]["objective"] for r in rows])
    agent = np.array([r["agent"]["metrics"]["objective"] for r in rows])
    tasks = sorted(set(r["family_id"] for r in rows))
    b = np.array(
        [
            np.median(
                [
                    r["baseline"]["metrics"]["objective"]
                    for r in rows
                    if r["family_id"] == t
                ]
            )
            for t in tasks
        ]
    )
    a = np.array(
        [
            np.median(
                [
                    r["agent"]["metrics"]["objective"]
                    for r in rows
                    if r["family_id"] == t
                ]
            )
            for t in tasks
        ]
    )
    paired = 100 * (b - a) / b
    rng = np.random.default_rng(4026)
    boot = np.median(paired[rng.integers(0, len(tasks), (5000, len(tasks)))], axis=1)
    return {
        "tasks": len(tasks),
        "seeds": sorted(set(r["seed"] for r in rows)),
        "median_objective_baseline": float(np.median(b)),
        "median_objective_agent": float(np.median(a)),
        "median_paired_improvement_percent": float(np.median(paired)),
        "paired_improvement_bootstrap_95_percent": np.quantile(
            boot, [0.025, 0.975]
        ).tolist(),
        "ratio_of_medians_improvement_percent": float(
            100 * (np.median(b) - np.median(a)) / np.median(b)
        ),
        "task_win_rate": float(np.mean(a < b)),
        "run_win_rate": float(np.mean(agent < baseline)),
        "feasibility_baseline": float(
            np.mean([r["baseline"]["metrics"]["feasible"] for r in rows])
        ),
        "feasibility_agent": float(
            np.mean([r["agent"]["metrics"]["feasible"] for r in rows])
        ),
        "median_within_task_seed_std_baseline": float(
            np.median(
                [
                    np.std(
                        [
                            r["baseline"]["metrics"]["objective"]
                            for r in rows
                            if r["family_id"] == t
                        ],
                        ddof=1,
                    )
                    for t in tasks
                ]
            )
        ),
        "median_within_task_seed_std_agent": float(
            np.median(
                [
                    np.std(
                        [
                            r["agent"]["metrics"]["objective"]
                            for r in rows
                            if r["family_id"] == t
                        ],
                        ddof=1,
                    )
                    for t in tasks
                ]
            )
        ),
    }


def benchmark(
    data_path, model_path, output, budget=192, n_tasks=24, seeds=(11, 22, 33)
):
    torch.set_num_threads(2)
    data = np.load(data_path)
    heldout = np.unique(data["family_id"][data["split"] == 2])
    if n_tasks > len(heldout):
        raise ValueError("not enough held-out families")
    families = np.random.default_rng(3026).permutation(heldout)[:n_tasks]
    model = load_model(model_path)
    rows = []
    start = time.perf_counter()
    for i, family in enumerate(families):
        task = data["family_tasks"][family]
        for seed in seeds:
            b = optimize(task, budget, seed)
            a = optimize(task, budget, seed, model)
            for result in (a, b):
                assert (
                    result["search_calls"] == budget and result["reporting_calls"] == 0
                )
                # High-density audit uses no feedback to either optimizer.
                metric = evaluate(
                    np.array(result["genes"])[None],
                    np.array(result["glasses"])[None],
                    task[None],
                    rings=8,
                    spokes=24,
                )
                result["dense_audit"] = {k: v[0].item() for k, v in metric.items()}
                result["audit_calls"] = 1
            rows.append(
                {"family_id": int(family), "seed": seed, "baseline": b, "agent": a}
            )
        print(f"benchmark task {i + 1}/{n_tasks}", flush=True)
    report = {
        "protocol": {
            "budget_per_method_per_task_seed": budget,
            "initial_candidates": 48,
            "refinement_batch": 24,
            "search_rays_per_design": 225,
            "audit_rays_per_design": 1737,
            "audit_calls_per_result": 1,
            "test_family_selection_seed": 3026,
            "training_seed": model.training_seed,
            "seeds": list(seeds),
            "family_ids": families.tolist(),
            "offline_training_designs": int(np.sum(data["split"] == 0)),
            "offline_validation_designs": int(np.sum(data["split"] == 1)),
            "offline_test_designs_not_used_by_search": int(np.sum(data["split"] == 2)),
            "budget_excludes_offline_training": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.perf_counter() - start,
        "summary": summarize(rows),
        "runs": rows,
    }
    for method in ("baseline", "agent"):
        report["summary"][f"dense_audit_feasibility_{method}"] = float(
            np.mean([r[method]["dense_audit"]["feasible"] for r in rows])
        )
        report["summary"][f"dense_audit_median_objective_{method}"] = float(
            np.median([r[method]["dense_audit"]["objective"] for r in rows])
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
