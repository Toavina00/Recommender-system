from typing import List, Tuple

import numpy as np

from .__base import update_user_bias as base_user_bias
from .__base import update_user_embedding as base_user_embedding
from .__variational import update_user_bias as var_user_bias
from .__variational import update_user_embedding as var_user_embedding


def new_user_base(
    movies_rating: List[Tuple[int, float]],
    movie_embedding: np.ndarray,
    movie_bias: np.ndarray,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    n_iter=3,
) -> Tuple[np.ndarray, np.ndarray]:
    embedding_dim = movie_embedding.shape[1]

    user_bias = np.random.randn(1)
    user_embedding = np.random.normal(0, np.sqrt(embedding_dim), (1, embedding_dim))

    movies = np.array([x[0] for x in movies_rating])
    ratings = np.array([x[1] for x in movies_rating])

    for _ in range(n_iter):
        base_user_bias(
            0,
            r_lambda,
            r_gamma,
            movies,
            ratings,
            user_embedding,
            user_bias,
            movie_embedding,
            movie_bias,
        )

        base_user_embedding(
            0,
            r_lambda,
            r_tau,
            movies,
            ratings,
            user_embedding,
            user_bias,
            movie_embedding,
            movie_bias,
        )

    return user_embedding, user_bias


def new_user_var(
    movies_rating: List[Tuple[int, float]],
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    n_iter=3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embedding_dim = movie_mean_embedding.shape[1]

    user_mean_bias = np.random.randn(1)
    user_var_bias = np.random.chisquare(1, (1,))
    user_mean_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (1, embedding_dim)
    )
    user_var_embedding = np.random.chisquare(embedding_dim, (1, embedding_dim))

    movies = np.array([x[0] for x in movies_rating])
    ratings = np.array([x[1] for x in movies_rating])

    for _ in range(n_iter):
        var_user_bias(
            0,
            r_lambda,
            r_gamma,
            movies,
            ratings,
            user_mean_embedding,
            user_mean_bias,
            user_var_bias,
            movie_mean_embedding,
            movie_mean_bias,
        )

        var_user_embedding(
            0,
            r_lambda,
            r_tau,
            movies,
            ratings,
            user_mean_embedding,
            user_var_embedding,
            user_mean_bias,
            movie_mean_embedding,
            movie_var_embedding,
            movie_mean_bias,
        )

    return user_mean_embedding, user_var_embedding, user_mean_bias, user_var_bias
