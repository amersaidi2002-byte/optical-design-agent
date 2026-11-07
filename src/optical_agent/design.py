"""Fixed topology: positive element, air gap, negative element, sensor.

The nine continuous genes in [0,1] encode four |R|/f, two thickness/f,
air gap/f, back focus/f, and clear semi-diameter / entrance pupil radius.
Two categorical genes select glasses. The aperture stop is at z=0.
"""

import numpy as np

LOW = np.array([0.55, 0.55, 0.8, 0.8, 0.025, 0.025, 0.025, 0.45, 1.05])
HIGH = np.array([3.0, 3.0, 5.0, 5.0, 0.12, 0.12, 0.22, 1.5, 1.8])
SIGNS = np.array([1.0, -1.0, -1.0, 1.0])


def decode(genes, tasks):
    genes = np.atleast_2d(np.asarray(genes, dtype=float))
    tasks = np.atleast_2d(np.asarray(tasks, dtype=float))
    if genes.shape[1] != 9 or tasks.shape[1] != 3:
        raise ValueError(
            "expected genes [N,9], tasks [N,3] = focal mm, half-FOV deg, f/#"
        )
    if len(genes) != len(tasks):
        raise ValueError("one task per design required")
    if not np.all(np.isfinite(genes)) or np.any((genes < 0) | (genes > 1)):
        raise ValueError("continuous genes must be finite and in [0,1]")
    if (
        not np.all(np.isfinite(tasks))
        or np.any(tasks <= 0)
        or np.any(tasks[:, 1] >= 40)
    ):
        raise ValueError("positive finite task values and half-FOV < 40° required")
    p = LOW + genes * (HIGH - LOW)
    f = tasks[:, 0]
    r = p[:, :4] * f[:, None] * SIGNS
    t1, t2, gap, back = (p[:, i] * f for i in (4, 5, 6, 7))
    vertices = np.stack([np.zeros_like(f), t1, t1 + gap, t1 + gap + t2], axis=1)
    aperture = f / (2 * tasks[:, 2])
    clear = aperture * p[:, 8]
    return r, vertices, vertices[:, -1] + back, aperture, clear


def random_designs(rng, n):
    return rng.random((n, 9)), rng.integers(0, 3, (n, 2))


def condition(tasks):
    """Fixed transforms, not fitted on test statistics."""
    return (np.asarray(tasks, dtype=np.float32) - [50, 6, 6]) / [30, 5, 3]


def prescription(genes, glasses, task):
    """Convert normalized genes into a readable physical prescription."""
    from .glass import NAMES

    radii, vertices, sensor, stop_radius, clear = decode(
        np.asarray(genes)[None], np.asarray(task)[None]
    )
    return {
        "surface_radii_mm": radii[0].tolist(),
        "surface_vertices_mm": vertices[0].tolist(),
        "element_thicknesses_mm": [
            float(vertices[0, 1] - vertices[0, 0]),
            float(vertices[0, 3] - vertices[0, 2]),
        ],
        "air_gap_mm": float(vertices[0, 2] - vertices[0, 1]),
        "back_focus_mm": float(sensor[0] - vertices[0, 3]),
        "sensor_z_mm": float(sensor[0]),
        "stop_radius_mm": float(stop_radius[0]),
        "clear_semi_diameter_mm": float(clear[0]),
        "glass_names": [NAMES[int(i)] for i in glasses],
    }
