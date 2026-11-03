import numpy as np

from bitcoin_aml.utils.seeding import make_rng, set_global_seed


def test_set_global_seed_reproduces_numpy_draws() -> None:
    set_global_seed(7)
    first = np.random.rand(5)

    set_global_seed(7)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_set_global_seed_different_seeds_diverge() -> None:
    set_global_seed(1)
    first = np.random.rand(5)

    set_global_seed(2)
    second = np.random.rand(5)

    assert not np.array_equal(first, second)


def test_make_rng_is_reproducible_and_independent() -> None:
    rng_a = make_rng(123)
    rng_b = make_rng(123)

    np.testing.assert_array_equal(
        rng_a.standard_normal(5),
        rng_b.standard_normal(5),
    )