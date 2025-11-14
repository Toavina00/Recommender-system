import os
from pathlib import Path

import numpy as np
from pytest import fixture

from utils.checkpoint import load_params, save_params


@fixture
def dummy_data():
    idx_to_id = ["a", "b", "c"]
    embedding = np.random.rand(3, 2)
    bias = np.random.rand(3)

    return (idx_to_id, embedding, bias)


def test_params_checkpoint(dummy_data):
    dummy_path = Path("dummy.save")

    if dummy_path.exists():
        os.remove(dummy_path.resolve())

    save_params(str(dummy_path.resolve()), dummy_data[0], dummy_data[1], dummy_data[2])

    if not dummy_path.exists():
        raise Exception("No save file created")

    idx_to_id, embedding, bias = load_params(str(dummy_path.resolve()))

    assert idx_to_id == dummy_data[0], "Idx to Id list inconsistent"
    assert (embedding == dummy_data[1]).all(), "Embedding array inconsistent"
    assert (bias == dummy_data[2]).all(), "Bias array inconsistent"

    os.remove(dummy_path.resolve())
