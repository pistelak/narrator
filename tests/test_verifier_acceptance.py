"""The acceptance bench must corrupt audio before calling a pass a false accept."""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

# Loaded by path, as test_intonation_probe does: `bench/` is a harness, not a
# package, and a bare `pytest` (CI) does not put the repo root on sys.path —
# `from bench.verifier_acceptance import ...` collected locally under
# `python -m pytest` and failed on both CI runners. The sys.modules entry is
# needed while the module executes because it defines a dataclass, and it is
# removed afterwards so no later import can receive this bench script.
_NAME = "_bench_verifier_acceptance_under_test"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, Path(__file__).resolve().parent.parent / "bench" / "verifier_acceptance.py"
)
_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = _module
try:
    _SPEC.loader.exec_module(_module)
finally:
    sys.modules.pop(_NAME, None)
corrupt = _module.corrupt


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
