"""The acceptance bench must corrupt audio before calling a pass a false accept."""

import random

import numpy as np
import pytest

from bench.verifier_acceptance import corrupt


@pytest.mark.parametrize("n_sentences", [2, 3, 4])
def test_swap_always_exchanges_two_adjacent_sentences(n_sentences: int) -> None:
    # Distinct constant-valued blocks make sentence order observable without ASR.
    # Seeds 0-19 previously left all two-sentence swaps and 5/20 others unchanged.
    audio = np.repeat(np.arange(n_sentences, dtype=np.float32), 100)
    possible = []
    for index in range(n_sentences - 1):
        order = list(range(n_sentences))
        order[index:index + 2] = order[index:index + 2][::-1]
        possible.append(np.repeat(np.array(order, dtype=np.float32), 100))
    for seed in range(20):
        swapped = corrupt(audio, "swap", n_sentences, random.Random(seed))
        assert any(np.array_equal(swapped, expected) for expected in possible)
