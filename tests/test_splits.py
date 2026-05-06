import numpy as np
import pytest

from data_utils.splits import make_split_indices


def test_split_indices_are_deterministic():
    first = make_split_indices(100, 50, 20, 30, seed=509)
    second = make_split_indices(100, 50, 20, 30, seed=509)

    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.val, second.val)
    np.testing.assert_array_equal(first.test, second.test)
    assert set(first.train).isdisjoint(set(first.test))


def test_set_seed_repeats_numpy_sequence():
    pytest.importorskip("torch")
    from utils import set_seed

    set_seed(123)
    first = np.random.rand(3)
    set_seed(123)
    second = np.random.rand(3)
    np.testing.assert_allclose(first, second)
