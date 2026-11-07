"""Budget-accounted generate–simulate–rank–refine search."""

import numpy as np

from .design import prescription, random_designs
from .physics import evaluate


def optimize(task, budget=192, seed=0, model=None, initial=48, batch=24):
    """Multi-start stochastic hill search; only the initial proposal differs.

    Both methods keep four elites, mutate continuous and categorical genes,
    and allocate 1/6 of each refinement batch to uniform restarts. There is
    no free simulator access, cached score, or test-set fitting.
    """
    if budget < initial or batch < 6:
        raise ValueError("budget >= initial and batch >= 6 required")
    rng = np.random.default_rng(seed)
    task = np.asarray(task, dtype=float)
    tasks = np.tile(task, (initial, 1))
    if model is None:
        x, g = random_designs(rng, initial)
    else:
        x, g = model.propose(tasks, rng)
    # Refinement receives the same random stream in paired runs.
    refine_rng = np.random.default_rng(seed + 1000003)
    scores = []
    used = 0
    history = []
    all_x = []
    all_g = []
    all_metrics = []
    elites_x = elites_g = elites_scores = None
    while used < budget:
        n = len(x)
        out = evaluate(x, g, np.tile(task, (n, 1)))
        all_x.extend(x)
        all_g.extend(g)
        scores.extend(out["objective"].tolist())
        all_metrics.extend({k: v[i].item() for k, v in out.items()} for i in range(n))
        used += n
        if elites_x is None:
            pool_x, pool_g, pool_s = x, g, out["objective"]
        else:
            pool_x = np.concatenate([elites_x, x])
            pool_g = np.concatenate([elites_g, g])
            pool_s = np.concatenate([elites_scores, out["objective"]])
        order = np.argsort(pool_s)[:4]
        elites_x, elites_g, elites_scores = pool_x[order], pool_g[order], pool_s[order]
        history.append({"calls": used, "objective": float(elites_scores[0])})
        if used == budget:
            break
        n = min(batch, budget - used)
        parent = refine_rng.integers(0, len(elites_x), n)
        scale = 0.16 * (1 - used / budget) + 0.025
        x = np.clip(elites_x[parent] + refine_rng.normal(0, scale, (n, 9)), 0, 1)
        g = elites_g[parent].copy()
        mutate = refine_rng.random(g.shape) < 0.12
        g[mutate] = refine_rng.integers(0, 3, mutate.sum())
        restarts = max(1, n // 6)
        x[-restarts:], g[-restarts:] = random_designs(refine_rng, restarts)
    best = int(np.argmin(scores))
    x = np.asarray(all_x)[best]
    g = np.asarray(all_g)[best]
    return {
        "prescription": prescription(x, g, task),
        "genes": x.tolist(),
        "glasses": g.tolist(),
        "task": task.tolist(),
        "metrics": all_metrics[best],
        "search_calls": used,
        "reporting_calls": 0,
        "history": history,
    }
