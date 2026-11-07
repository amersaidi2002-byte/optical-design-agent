import numpy as np

from optical_agent.design import decode
from optical_agent.glass import index
from optical_agent.physics import clearance, evaluate, paraxial, refract, trace


def test_catalog_refractive_indices_and_normal_dispersion():
    np.testing.assert_allclose(
        index(np.arange(3), 0.5876), [1.51680, 1.62005, 1.67271], atol=5e-5
    )
    assert np.all(index(np.arange(3), 0.4861) > index(np.arange(3), 0.6563))


def test_snell_law_oblique_incidence():
    theta = np.deg2rad(30)
    d = np.array([[np.sin(theta), 0, np.cos(theta)]])
    out, tir = refract(d, np.array([[0, 0, -1.0]]), np.array([1.0]), np.array([1.5]))
    assert not tir[0]
    np.testing.assert_allclose(out[0, 0], np.sin(theta) / 1.5, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1, atol=1e-12)


def test_total_internal_reflection():
    d = np.array([[np.sin(np.deg2rad(60)), 0, np.cos(np.deg2rad(60))]])
    _, tir = refract(d, np.array([[0, 0, -1.0]]), np.array([1.5]), np.array([1.0]))
    assert tir[0]


def test_thick_lens_lensmaker_limit():
    n = float(index(0, 0.5876))
    radii = np.array([[100.0, -100.0, -1e12, 1e12]])
    vertices = np.array([[0.0, 5.0, 10.0, 15.0]])
    efl, _ = paraxial(radii, vertices, np.array([[0, 0]]))
    expected = 1 / ((n - 1) * (1 / 100 - 1 / (-100) + (n - 1) * 5 / (n * 100 * (-100))))
    np.testing.assert_allclose(efl[0], expected, rtol=1e-8)


def test_full_ray_trace_matches_independent_paraxial_focus():
    x = np.array([[0.1, 0.1, 0.8, 0.8, 0.5, 0.5, 0.2, 0.5, 0.4]])
    g = np.array([[0, 1]])
    task = np.array([[50.0, 0.001, 1000.0]])
    r, z, *_ = decode(x, task)
    efl, bfl = paraxial(r, z, g)
    _, valid, paths = trace(x, g, task, keep_rays=True)
    assert valid.all()
    exit_point = paths[-2][0, 0, 1, 1]
    sensor_point = paths[-1][0, 0, 1, 1]
    slope = (sensor_point[0] - exit_point[0]) / (sensor_point[2] - exit_point[2])
    actual_focus = exit_point[2] - exit_point[0] / slope
    np.testing.assert_allclose(actual_focus, z[0, -1] + bfl[0], rtol=1e-5)
    # Correct branch for negative R: cap hit is near its vertex, not far side of sphere.
    for s in range(4):
        assert abs(paths[s + 1][0, 0, 1, 1, 2] - z[0, s]) < 0.001


def test_clearance_rejects_colliding_elements():
    r = np.array([[10.0, -10.0, -20.0, 20.0]])
    z = np.array([[0.0, 0.1, 1.0, 1.1]])
    assert clearance(r, z, np.array([4.0]))[0] < 0


def test_batch_and_single_results_identical():
    rng = np.random.default_rng(9)
    x = rng.random((8, 9))
    g = rng.integers(0, 3, (8, 2))
    task = np.tile([50, 6, 5], (8, 1))
    batch = evaluate(x, g, task)
    for i in range(8):
        single = evaluate(x[i : i + 1], g[i : i + 1], task[i : i + 1])
        for key in batch:
            np.testing.assert_allclose(batch[key][i], single[key][0], rtol=1e-10)
