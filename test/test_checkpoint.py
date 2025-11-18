import os
from pathlib import Path

import numpy as np
from pytest import fixture

from utils.checkpoint import load_params, save_params


@fixture
def dummy_data():
    r_lambda = 0.1
    r_tau = 0.1
    r_gamma = 0.1
    u_idx_to_id = ["a", "b", "c"]
    m_idx_to_id = ["a", "b", "c"]
    u_bias = np.random.rand(3)
    m_bias = np.random.rand(3)
    u_embedding = np.random.rand(3, 2)
    m_embedding = np.random.rand(3, 2)
    f_embedding = np.random.rand(3, 2)

    return (
        r_lambda,
        r_tau,
        r_gamma,
        u_idx_to_id,
        m_idx_to_id,
        u_bias,
        m_bias,
        u_embedding,
        m_embedding,
        None,
        f_embedding,
    )


def test_params_checkpoint(dummy_data):
    dummy_path = Path("dummy.npz")

    if dummy_path.exists():
        os.remove(dummy_path.resolve())

    save_params(str(dummy_path.resolve()), *dummy_data)

    if not dummy_path.exists():
        raise Exception("No save file created")

    params = load_params(str(dummy_path.resolve()))

    assert params["r_lambda"] == dummy_data[0], "Checkpoint Inconsistent"
    assert params["r_tau"] == dummy_data[1], "Checkpoint Inconsistent"
    assert params["r_gamma"] == dummy_data[2], "Checkpoint Inconsistent"
    assert params["idx_to_user_id"] == dummy_data[3], "Checkpoint Inconsistent"
    assert params["idx_to_movie_id"] == dummy_data[4], "Checkpoint Inconsistent"
    assert (params["user_bias"] == dummy_data[5]).all(), "Checkpoint Inconsistent"
    assert (params["movie_bias"] == dummy_data[6]).all(), "Checkpoint Inconsistent"
    assert (params["user_embedding"] == dummy_data[7]).all(), "Checkpoint Inconsistent"
    assert (params["movie_embedding"] == dummy_data[8]).all(), "Checkpoint Inconsistent"
    assert len(params["idx_to_feat_id"]) == 0, "Checkpoint Inconsistent"
    assert (params["feat_embedding"] == dummy_data[10]).all(), "Checkpoint Inconsistent"

    os.remove(dummy_path.resolve())
