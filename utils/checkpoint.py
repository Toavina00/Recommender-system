import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def save_params(
    fname: str,
    r_lambda: float,
    r_tau: float,
    r_gamma: float,
    idx_to_user_id: List[Any],
    idx_to_movie_id: List[Any],
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embedding: np.ndarray,
    movie_embedding: np.ndarray,
    idx_to_feat_id: List[Any] | None = None,
    feat_embedding: np.ndarray | None = None,
):
    fpath = Path(fname)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    if fpath.exists():
        shutil.rmtree(fpath)

    np.savez(
        fpath.resolve(),
        reg=np.array([r_lambda, r_tau, r_gamma]),
        idx_to_user_id=np.array(idx_to_user_id),
        idx_to_movie_id=np.array(idx_to_movie_id),
        user_embedding=user_embedding,
        movie_embedding=movie_embedding,
        user_bias=user_bias,
        movie_bias=movie_bias,
        idx_to_feat_id=idx_to_feat_id if idx_to_feat_id is not None else np.array([]),
        feat_embedding=feat_embedding if feat_embedding is not None else np.array([]),
    )


def load_params(fname: str) -> Dict:
    fpath = Path(fname)

    loaded = np.load(fpath)

    params = dict()
    params["r_lambda"] = loaded["reg"][0]
    params["r_tau"] = loaded["reg"][1]
    params["r_gamma"] = loaded["reg"][2]
    params["idx_to_user_id"] = loaded["idx_to_user_id"].tolist()
    params["idx_to_movie_id"] = loaded["idx_to_movie_id"].tolist()
    params["user_bias"] = loaded["user_bias"]
    params["movie_bias"] = loaded["movie_bias"]
    params["user_embedding"] = loaded["user_embedding"]
    params["movie_embedding"] = loaded["movie_embedding"]
    params["idx_to_feat_id"] = loaded["idx_to_feat_id"].tolist()
    params["feat_embedding"] = loaded["feat_embedding"]

    return params
