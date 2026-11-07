import numpy as np
import pytest

from optical_agent.dataset import elite_indices, generate
from optical_agent.model import ConditionalVAE
from optical_agent.search import optimize


def test_family_partition_and_elite_isolation(tmp_path):
    path = tmp_path / "designs.npz"
    generate(path, families=30, members=5, seed=1)
    data = np.load(path)
    families = [set(data["family_id"][data["split"] == s]) for s in range(3)]
    assert all(families[i].isdisjoint(families[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    assert len(data["genes"]) == 150
    for s in (0, 1, 2):
        assert np.all(data["split"][elite_indices(data, s)] == s)


def test_search_budget_is_exact_and_best_is_monotonic(monkeypatch):
    import optical_agent.search as search

    original = search.evaluate
    calls = []

    def counted(x, g, t):
        calls.append(len(x))
        return original(x, g, t)

    monkeypatch.setattr(search, "evaluate", counted)
    result = optimize([50, 5, 6], budget=113, seed=2)
    assert sum(calls) == 113 == result["search_calls"]
    assert result["reporting_calls"] == 0
    assert np.all(np.diff([v["objective"] for v in result["history"]]) <= 0)
    assert result["history"][-1]["calls"] == 113


def test_search_is_reproducible_and_model_proposals_are_valid():
    a = optimize([50, 5, 6], budget=48, seed=3)
    b = optimize([50, 5, 6], budget=48, seed=3)
    assert a == b
    model = ConditionalVAE()
    x, g = model.propose(np.tile([50, 5, 6], (7, 1)), np.random.default_rng(1))
    assert x.shape == (7, 9) and g.shape == (7, 2)
    assert np.all((x >= 0) & (x <= 1)) and np.all((g >= 0) & (g <= 2))


def test_invalid_task_rejected():
    with pytest.raises(ValueError):
        optimize([-1, 5, 6], budget=48)
