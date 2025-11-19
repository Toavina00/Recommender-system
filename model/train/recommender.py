from typing import List, Tuple

import numba as nb
import numpy as np


@nb.njit
def compute_loss(
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    movie_users: List[Tuple[np.ndarray, np.ndarray]],
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
) -> float:
    loss = 0.0

    for movie_idx in nb.prange(len(movie_users)):
        users, ratings = movie_users[movie_idx]

        if len(ratings) == 0:
            continue

        e_vec = ratings - (
            user_embeddings[users] @ movie_embeddings[movie_idx]
            + user_bias[users]
            + movie_bias[movie_idx]
        )
        e_sum = e_vec @ e_vec
        loss += e_sum

    loss *= r_lambda * 0.5
    loss += r_gamma * 0.5 * (user_bias @ user_bias)
    loss += r_gamma * 0.5 * (movie_bias @ movie_bias)
    loss += r_tau * 0.5 * np.sum(user_embeddings * user_embeddings)
    loss += r_tau * 0.5 * np.sum(movie_embeddings * movie_embeddings)

    return loss


@nb.njit
def compute_rmse(
    user_movies: List[Tuple[np.ndarray, np.ndarray]],
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
) -> float:
    error, total = 0.0, 0.0
    for i in range(len(user_movies)):
        movies, ratings = user_movies[i]

        if len(ratings) == 0:
            continue

        e_vec = ratings - (
            movie_embeddings[movies] @ user_embeddings[i]
            + movie_bias[movies]
            + user_bias[i]
        )
        e_sum = e_vec @ e_vec
        error += e_sum
        total += len(ratings)

    error /= total
    error = np.sqrt(error)

    return error


@nb.njit(parallel=True)
def optimize_users(
    train_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    embedding_dim: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
) -> None:
    for user_idx in nb.prange(len(train_user_movies)):
        movies, ratings = train_user_movies[user_idx]

        if len(ratings) == 0:
            continue

        # Bias Update
        new_user_bias = np.sum(
            ratings
            - (
                movie_embeddings[movies] @ user_embeddings[user_idx]
                + movie_bias[movies]
            )
        )
        new_user_bias *= r_lambda
        new_user_bias /= r_lambda * len(ratings) + r_gamma

        user_bias[user_idx] = new_user_bias

        # Embedding update
        movie_mat = movie_embeddings[movies].T @ movie_embeddings[movies]
        movie_vec = movie_embeddings[movies].T @ (
            ratings - movie_bias[movies] - user_bias[user_idx]
        )

        movie_mat *= r_lambda
        movie_mat += r_tau * np.eye(embedding_dim)
        movie_vec *= r_lambda

        inv_movie_mat = np.linalg.inv(movie_mat)
        new_user_embedding = inv_movie_mat @ movie_vec

        user_embeddings[user_idx] = new_user_embedding


@nb.njit(parallel=True)
def optimize_movie(
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    embedding_dim: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
) -> None:
    for movie_idx in nb.prange(len(train_movie_users)):
        users, ratings = train_movie_users[movie_idx]

        if len(ratings) == 0:
            continue

        # Bias Update
        new_movie_bias = np.sum(
            ratings
            - (user_embeddings[users] @ movie_embeddings[movie_idx] + user_bias[users])
        )
        new_movie_bias *= r_lambda
        new_movie_bias /= r_lambda * len(ratings) + r_gamma

        movie_bias[movie_idx] = new_movie_bias

        # Embedding update
        users_mat = user_embeddings[users].T @ user_embeddings[users]
        user_vec = user_embeddings[users].T @ (
            ratings - user_bias[users] - movie_bias[movie_idx]
        )

        users_mat *= r_lambda
        users_mat += r_tau * np.eye(embedding_dim)
        user_vec *= r_lambda

        inv_user_mat = np.linalg.inv(users_mat)
        new_movie_embedding = inv_user_mat @ user_vec

        movie_embeddings[movie_idx] = new_movie_embedding


@nb.njit
def update_params(
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    train_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    embedding_dim: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
):
    optimize_users(
        train_user_movies,
        embedding_dim,
        r_lambda,
        r_gamma,
        r_tau,
        user_bias,
        movie_bias,
        user_embeddings,
        movie_embeddings,
    )
    optimize_movie(
        train_movie_users,
        embedding_dim,
        r_lambda,
        r_gamma,
        r_tau,
        user_bias,
        movie_bias,
        user_embeddings,
        movie_embeddings,
    )


@nb.njit
def training_loop(
    train_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    val_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    val_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    embedding_dim: int = 2,
    r_lambda: float = 0.05,
    r_gamma: float = 0.05,
    r_tau: float = 0.05,
    n_iter: int = 10,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_loss = np.zeros((n_iter))
    train_rmse = np.zeros((n_iter))
    val_loss = np.zeros((n_iter))
    val_rmse = np.zeros((n_iter))

    n_user, n_movie = (
        len(train_user_movies),
        len(train_movie_users),
    )

    user_bias = np.random.randn(n_user)
    movie_bias = np.random.randn(n_movie)
    user_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    movie_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    )

    for iter in range(n_iter):
        update_params(
            train_movie_users,
            train_user_movies,
            embedding_dim,
            r_lambda,
            r_gamma,
            r_tau,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
        )

        train_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            train_movie_users,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
        )
        train_rmse[iter] = compute_rmse(
            train_user_movies, user_bias, movie_bias, user_embeddings, movie_embeddings
        )
        val_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            val_movie_users,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
        )
        val_rmse[iter] = compute_rmse(
            val_user_movies, user_bias, movie_bias, user_embeddings, movie_embeddings
        )

    return (
        train_loss,
        train_rmse,
        val_loss,
        val_rmse,
        user_bias,
        movie_bias,
        user_embeddings,
        movie_embeddings,
    )
