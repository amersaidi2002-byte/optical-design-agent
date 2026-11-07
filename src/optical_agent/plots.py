"""Scientific plots generated only from measured results."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .design import decode
from .glass import NAMES
from .physics import trace

COLORS = ["#268bd2", "#d6a83a", "#d05355"]


def design_plot(result, path):
    x = np.array(result["genes"])[None]
    g = np.array(result["glasses"])[None]
    task = np.array(result["task"])[None]
    r, z, sensor, ap, clear = decode(x, task)
    intercept, valid, paths = trace(x, g, task, keep_rays=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    ax = axes[0]
    h = np.linspace(-clear[0], clear[0], 200)
    for pair in (0, 2):
        curves = []
        for s in (pair, pair + 1):
            radius = r[0, s]
            curves.append(
                z[0, s] + radius - np.sign(radius) * np.sqrt(radius**2 - h**2)
            )
        ax.fill_betweenx(h, curves[0], curves[1], color="#91c8d2", alpha=0.5)
        ax.plot(curves[0], h, color="#296d7c", lw=1.2)
        ax.plot(curves[1], h, color="#296d7c", lw=1.2)
        ax.text(
            np.mean(z[0, pair : pair + 2]),
            -clear[0] - 2,
            NAMES[g[0, pair // 2]],
            ha="center",
            fontsize=8,
        )
    ax.axvline(sensor[0], color="#536878", lw=2, label="Sensor")
    paths = np.stack(paths)
    # Meridional-like subset: centre and first sample on each pupil ring.
    for field in (0, 2):
        for ray in (0, 1, 9, 17):
            if valid[0, field, 1, ray]:
                p = paths[:, 0, field, 1, ray]
                ax.plot(p[:, 2], p[:, 0], color=COLORS[field], lw=0.8, alpha=0.7)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set(
        xlabel="Optical axis z (mm)",
        ylabel="Ray height x (mm)",
        title="Sequential spherical ray tracing",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax = axes[1]
    for wave in range(3):
        pts = intercept[0, 2, wave, :, :2][valid[0, 2, wave]]
        if len(pts):
            # Subtract the same polychromatic centroid for every colour.
            allpts = intercept[0, 2, :, :, :2][valid[0, 2]]
            pts = (pts - allpts.mean(axis=0)) * 1000
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=11,
                color=COLORS[wave],
                alpha=0.8,
                label=["486.1 nm", "587.6 nm", "656.3 nm"][wave],
            )
    ax.set(
        xlabel="x − centroid (µm)",
        ylabel="y − centroid (µm)",
        title="Edge-field polychromatic spot",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"f = {task[0, 0]:.1f} mm  ·  half-FOV = {task[0, 1]:.1f}°  ·  f/{task[0, 2]:.1f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(report_path, training_path, output):
    report = json.loads(Path(report_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = report["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for method, color, label in [
        ("baseline", "#8b939e", "Multi-start"),
        ("agent", "#087f8c", "Conditional CVAE"),
    ]:
        histories = np.array(
            [[v["objective"] for v in r[method]["history"]] for r in rows]
        )
        calls = [v["calls"] for v in rows[0][method]["history"]]
        q = np.quantile(histories, [0.25, 0.5, 0.75], axis=0)
        axes[0].plot(calls, q[1], color=color, label=label)
        axes[0].fill_between(calls, q[0], q[2], color=color, alpha=0.15)
    axes[0].set(
        xlabel="Simulator calls / search",
        ylabel="Normalized objective",
        title="Median and interquartile range",
    )
    axes[0].legend(fontsize=8)
    families = sorted(set(r["family_id"] for r in rows))
    b = [
        np.median(
            [r["baseline"]["metrics"]["objective"] for r in rows if r["family_id"] == t]
        )
        for t in families
    ]
    a = [
        np.median(
            [r["agent"]["metrics"]["objective"] for r in rows if r["family_id"] == t]
        )
        for t in families
    ]
    axes[1].scatter(b, a, color="#087f8c", s=28)
    lim = max(b + a) * 1.05
    axes[1].plot([0, lim], [0, lim], "--", color="gray", lw=1)
    axes[1].set(
        xlabel="Multi-start objective",
        ylabel="CVAE objective",
        title="Held-out tasks · median over 3 seeds",
        xlim=(0, lim),
        ylim=(0, lim),
    )
    history = json.loads(Path(training_path).read_text())["history"]
    for key, label in [
        ("train_loss", "Train"),
        ("validation_loss", "Validation (mean latent)"),
    ]:
        axes[2].plot(
            [r["epoch"] for r in history], [r[key] for r in history], label=label
        )
    axes[2].set(xlabel="Epoch", ylabel="CVAE loss", title="Training diagnostics")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "benchmark.png", dpi=180)
    plt.close(fig)
    # Fixed first task/seed, never cherry-picked by performance.
    design_plot(rows[0]["agent"], output / "design.png")
    s = report["summary"]
    text = [
        "# Measured experiment results",
        "",
        f"{s['tasks']} held-out families; seeds {s['seeds']}; {report['protocol']['budget_per_method_per_task_seed']} search calls per method and run.",
        "Offline data-generation costs are excluded from the online comparison. The baseline receives no learned prior.",
        "",
        "| Metric | Multi-start | Conditional CVAE |",
        "|---|---:|---:|",
        f"| Median objective (per-task median over seeds) | {s['median_objective_baseline']:.4f} | {s['median_objective_agent']:.4f} |",
        f"| Feasible runs (search quadrature) | {s['feasibility_baseline']:.1%} | {s['feasibility_agent']:.1%} |",
        f"| Feasible runs (dense independent audit) | {s['dense_audit_feasibility_baseline']:.1%} | {s['dense_audit_feasibility_agent']:.1%} |",
        f"| Median within-task objective std across seeds | {s['median_within_task_seed_std_baseline']:.4f} | {s['median_within_task_seed_std_agent']:.4f} |",
        "",
        f"Median paired task improvement: **{s['median_paired_improvement_percent']:.2f}%**.",
        f"Task bootstrap 95% interval: {s['paired_improvement_bootstrap_95_percent'][0]:.2f}% to {s['paired_improvement_bootstrap_95_percent'][1]:.2f}% (5,000 resamples; fixed model seed).",
        f"Ratio-of-medians improvement: {s['ratio_of_medians_improvement_percent']:.2f}%. These are different estimands.",
        f"Agent wins on {s['task_win_rate']:.1%} of tasks, using task medians.",
        "",
        "## All tasks",
        "",
        "| Family | f (mm) | Half-FOV (°) | f/# | Multi-start median | CVAE median | Paired improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, bv, av in zip(families, b, a):
        task = next(r for r in rows if r["family_id"] == family)["agent"]["task"]
        text.append(
            f"| {family} | {task[0]:.2f} | {task[1]:.2f} | {task[2]:.2f} | {bv:.4f} | {av:.4f} | {100 * (bv - av) / bv:.1f}% |"
        )
    text += [
        "",
        "## Interpretation",
        "",
        "This is a synthetic, fixed-topology proof of concept. Family grouping prevents related prescription variants crossing splits, but does not establish generalization to new topology, glass catalogs, or real manufactured optics.",
        "The three seeds vary online proposals/refinement only. There is one dataset seed and one training seed; the confidence interval does not include model-training uncertainty.",
        "Transmission means the fraction of geometrically valid sampled rays, not optical energy throughput. Feasibility is sampled and is not a manufacturing certification.",
        "Dense audit scores use 1,737 rays per design after optimization; these calls never feed back into the search. Both methods receive the same one-call audit.",
        "No 27% target was imposed or copied from the project brief. The numbers above are computed from benchmark.json.",
        "",
    ]
    (output.parent / "results.md").write_text("\n".join(text))
