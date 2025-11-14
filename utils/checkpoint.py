from typing import Any, List, Tuple

import numpy as np


def save_params(
    fname: str, idx_to_id: List[Any], embedding: np.ndarray, bias: np.ndarray
) -> None:
    with open(fname, "x") as f:
        for i in range(len(idx_to_id)):
            id = idx_to_id[i]
            emb = " ".join([f"{k:.18e}" for k in embedding[i]])
            b = f"{bias[i]:.18e}"
            f.write(f"{id},{emb},{b}\n")


def load_params(fname: str) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    idx_to_id, embedding, bias = [], [], []
    with open(fname, "r") as f:
        while line := f.readline():
            id, emb, b = line.split(",")
            idx_to_id.append(id)
            embedding.append([float(x) for x in emb.split(" ")])
            bias.append(float(b))

    return idx_to_id, np.array(embedding), np.array(bias)
