"""Vectorized sequential spherical ray tracer (mm, radians internally)."""

import numpy as np

from .design import decode
from .glass import WAVELENGTHS, index


def refract(direction, normal, n_in, n_out):
    """Snell refraction; normal is oriented against incident propagation.

    Returns a finite unit direction and a total-internal-reflection mask.
    Inputs broadcast over all axes before the last vector dimension.
    """
    cosi = -np.sum(direction * normal, axis=-1)
    normal = np.where((cosi < 0)[..., None], -normal, normal)
    cosi = np.abs(cosi)
    eta = np.asarray(n_in) / np.asarray(n_out)
    k = 1 - eta**2 * (1 - cosi**2)
    out = (
        eta[..., None] * direction
        + (eta * cosi - np.sqrt(np.maximum(k, 0)))[..., None] * normal
    )
    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-15)
    return out, k < 0


def paraxial(radii, vertices, glasses, wavelength=0.5876):
    """ABCD in [height, n*angle]; return effective and back focal lengths."""
    n = len(radii)
    media = np.stack(
        [
            np.ones(n),
            index(glasses[:, 0], wavelength),
            np.ones(n),
            index(glasses[:, 1], wavelength),
            np.ones(n),
        ],
        axis=1,
    )
    matrix = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    for s in range(4):
        if s:
            trans = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
            trans[:, 0, 1] = (vertices[:, s] - vertices[:, s - 1]) / media[:, s]
            matrix = trans @ matrix
        surf = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
        surf[:, 1, 0] = -(media[:, s + 1] - media[:, s]) / radii[:, s]
        matrix = surf @ matrix
    c = matrix[:, 1, 0]
    c = np.where(np.abs(c) < 1e-12, 1e-12, c)
    return -1 / c, -matrix[:, 0, 0] / c


def pupil_points(rings=3, spokes=8):
    """Equal-area annular samples plus centre, deterministic disk quadrature."""
    if rings < 1 or spokes < 4:
        raise ValueError("rings >= 1 and spokes >= 4 required")
    pts = [[0.0, 0.0]]
    for j in range(rings):
        radius = np.sqrt((j + 0.5) / rings)
        for angle in np.arange(spokes) * (2 * np.pi / spokes) + j * 0.37:
            pts.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.asarray(pts)


def clearance(radii, vertices, clear):
    """Minimum axial material/air clearance across the shared clear disk.

    All surfaces are spherical caps; pairwise sag differences are monotone
    for this fixed sign pattern. Centre and edge therefore bound clearance.
    """
    cap_ok = np.all(np.abs(radii) > clear[:, None], axis=1)
    sag = radii - np.sign(radii) * np.sqrt(
        np.maximum(radii**2 - clear[:, None] ** 2, 0)
    )
    edge = vertices + sag
    minimum = np.minimum(np.diff(vertices, axis=1), np.diff(edge, axis=1)).min(axis=1)
    return np.where(cap_ok, minimum, -1e3)


def trace(genes, glasses, tasks, rings=3, spokes=8, keep_rays=False):
    """Trace 3 fields x 3 wavelengths x sampled pupil rays per design.

    Returns sensor intercepts, validity and optional vertex-to-vertex paths.
    Surface branch chosen by curvature sign, with forward propagation checks.
    """
    r, z, sensor, aperture, clear = decode(genes, tasks)
    glasses = np.atleast_2d(np.asarray(glasses))
    if glasses.shape != (len(r), 2):
        raise ValueError("expected glasses [N,2]")
    n = len(r)
    pupil = pupil_points(rings, spokes)
    shape = (n, 3, 3, len(pupil))
    theta = np.deg2rad(np.asarray(tasks)[:, 1, None] * [0.0, 0.7, 1.0])
    direction = np.zeros(shape + (3,))
    direction[..., 0] = np.sin(theta)[:, :, None, None]
    direction[..., 2] = np.cos(theta)[:, :, None, None]
    origin = np.zeros_like(direction)
    origin[..., :2] = aperture[:, None, None, None, None] * pupil[None, None, None]
    origin[..., 2] = -2 * aperture[:, None, None, None]
    origin[..., 0] += origin[..., 2] * np.tan(theta)[:, :, None, None]
    valid = np.ones(shape, dtype=bool)
    paths = [origin.copy()] if keep_rays else None
    ns = [
        np.ones((n, 3)),
        index(glasses[:, 0, None], WAVELENGTHS[None]),
        np.ones((n, 3)),
        index(glasses[:, 1, None], WAVELENGTHS[None]),
        np.ones((n, 3)),
    ]
    for s in range(4):
        center = np.zeros((n, 1, 1, 1, 3))
        center[..., 2] = (z[:, s] + r[:, s])[:, None, None, None]
        oc = origin - center
        b = np.sum(oc * direction, axis=-1)
        c = np.sum(oc * oc, axis=-1) - r[:, s, None, None, None] ** 2
        disc = b * b - c
        step = -b - np.sign(r[:, s, None, None, None]) * np.sqrt(np.maximum(disc, 0))
        hit = origin + step[..., None] * direction
        valid &= (disc >= 0) & (step > 1e-9)
        valid &= np.sum(hit[..., :2] ** 2, axis=-1) <= clear[:, None, None, None] ** 2
        normal = (hit - center) / np.abs(r[:, s, None, None, None, None])
        direction, tir = refract(
            direction, normal, ns[s][:, None, :, None], ns[s + 1][:, None, :, None]
        )
        valid &= ~tir & (direction[..., 2] > 1e-8)
        origin = hit
        if keep_rays:
            paths.append(hit.copy())
    step = (sensor[:, None, None, None] - origin[..., 2]) / np.maximum(
        direction[..., 2], 1e-8
    )
    valid &= step > 0
    intercept = origin + step[..., None] * direction
    if keep_rays:
        paths.append(intercept.copy())
    return intercept, valid, paths


def evaluate(genes, glasses, tasks, rings=3, spokes=8):
    """One simulator call per design; all metrics use the same ray sample."""
    tasks = np.atleast_2d(np.asarray(tasks))
    r, z, sensor, aperture, clear = decode(genes, tasks)
    intercept, valid, _ = trace(genes, glasses, tasks, rings, spokes)
    # Common polychromatic centroid within each field retains lateral colour.
    weights = valid[..., None]
    count = valid.sum(axis=(2, 3))
    centroid = (intercept[..., :2] * weights).sum(axis=(2, 3)) / np.maximum(
        count[..., None], 1
    )
    squared = np.sum((intercept[..., :2] - centroid[:, :, None, None]) ** 2, axis=-1)
    field_rms = np.sqrt((squared * valid).sum(axis=(2, 3)) / np.maximum(count, 1))
    field_rms = np.where(count >= 3, field_rms, tasks[:, 0, None])
    rms = np.sqrt(np.mean(field_rms**2, axis=1))
    efl, bfl = paraxial(r, z, np.asarray(glasses))
    focal_error = np.abs(efl - tasks[:, 0]) / tasks[:, 0]
    transmission = valid.mean(axis=(1, 2, 3))
    min_clearance = clearance(r, z, clear)
    # Require >= 0.5 mm material / inter-element clearance, and 90% valid rays
    # per field/wavelength, not merely on aggregate.
    worst_transmission = valid.mean(axis=3).min(axis=(1, 2))
    feasible = (
        (min_clearance >= 0.5)
        & (worst_transmission >= 0.9)
        & (focal_error <= 0.1)
        & (efl > 0)
    )
    spot_term = rms / (0.01 * tasks[:, 0])
    clear_term = np.maximum(0.5 - min_clearance, 0) / (0.01 * tasks[:, 0])
    objective = (
        spot_term
        + 5 * focal_error
        + 10 * (1 - transmission)
        + 2 * clear_term
        + 10 * (efl <= 0)
    )
    return {
        "objective": objective,
        "rms_spot_mm": rms,
        "efl_mm": efl,
        "focal_error": focal_error,
        "transmission": transmission,
        "worst_transmission": worst_transmission,
        "min_clearance_mm": min_clearance,
        "feasible": feasible,
    }
