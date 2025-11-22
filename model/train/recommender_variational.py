from typing import List, Tuple

import numba as nb
import numpy as np


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


@nb.njit
def compute_elbo(
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_movies: List[Tuple[np.ndarray, np.ndarray]],
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    user_var_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
    movie_var_bias: np.ndarray,
):
    elbo = 0.0

    for user_idx in range(len(user_movies)):
        movies, ratings = user_movies[user_idx]

        if len(ratings) == 0:
            continue

        elbo += ratings.T @ ratings
        elbo -= 2.0 * (ratings.T @ movie_mean_bias[movies])
        elbo -= (
            2.0
            * (ratings - movie_mean_bias[movies] - user_mean_bias[user_idx])
            @ (movie_mean_embedding[movies] @ user_mean_embedding[user_idx])
        )
        elbo -= (
            2.0 * np.sum(ratings - movie_mean_bias[movies]) * user_mean_bias[user_idx]
        )

        mat = (
            np.diag(movie_var_embedding[movies].sum(axis=0))
            + movie_mean_embedding[movies].T @ movie_mean_embedding[movies]
        )
        mat = (
            np.diag(user_var_embedding[user_idx])
            + np.outer(user_mean_embedding[user_idx], user_mean_embedding[user_idx])
        ) @ mat

        elbo += np.sum(np.diag(mat))

    elbo *= -0.5 * r_lambda
    elbo -= 0.5 * r_tau * np.sum(user_mean_embedding * user_mean_embedding)
    elbo -= 0.5 * r_tau * np.sum(movie_mean_embedding * movie_mean_embedding)
    elbo -= 0.5 * r_tau * np.sum(user_var_embedding)
    elbo -= 0.5 * r_tau * np.sum(movie_var_embedding)
    elbo -= 0.5 * (r_gamma + r_lambda) * np.sum(user_mean_bias**2 + user_var_bias)
    elbo -= 0.5 * (r_gamma + r_lambda) * np.sum(movie_mean_bias**2 + movie_var_bias)

    return elbo


@nb.njit
def update_users(
    user_idx: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_movies: List[Tuple[np.ndarray, np.ndarray]],
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    user_var_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
):
    embedding_dim = user_mean_embedding.shape[1]

    movies, ratings = user_movies[user_idx]

    if len(ratings) == 0:
        return

    # Bias Update
    new_user_mean_bias = np.sum(
        ratings
        - (
            movie_mean_embedding[movies] @ user_mean_embedding[user_idx]
            + movie_mean_bias[movies]
        )
    )
    new_user_mean_bias *= r_lambda
    new_user_mean_bias /= r_lambda * len(ratings) + r_gamma

    user_var_bias[user_idx] = 1.0 / (r_lambda * len(ratings) + r_gamma)
    user_mean_bias[user_idx] = new_user_mean_bias

    # Embedding update
    movie_mat = movie_mean_embedding[movies].T @ movie_mean_embedding[movies]
    movie_mat += np.diag(movie_var_embedding[movies].sum(axis=0))
    movie_mat *= r_lambda
    movie_mat += r_tau * np.eye(embedding_dim)

    movie_vec = movie_mean_embedding[movies].T @ (
        ratings - movie_mean_bias[movies] - movie_mean_bias[user_idx]
    )
    movie_vec *= r_lambda

    inv_movie_mat = 1.0 / np.diag(movie_mat)
    new_user_mean_embedding = np.diag(inv_movie_mat) @ movie_vec
    new_user_var_embedding = inv_movie_mat

    user_var_embedding[user_idx] = new_user_var_embedding
    user_mean_embedding[user_idx] = new_user_mean_embedding


@nb.njit
def update_movie(
    movie_idx: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    movie_users: List[Tuple[np.ndarray, np.ndarray]],
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
    movie_var_bias: np.ndarray,
):
    embedding_dim = user_mean_embedding.shape[1]

    users, ratings = movie_users[movie_idx]

    if len(ratings) == 0:
        return

    # Bias Update
    new_movie_mean_bias = np.sum(
        ratings
        - (
            user_mean_embedding[users] @ movie_mean_embedding[movie_idx]
            + user_mean_bias[users]
        )
    )
    new_movie_mean_bias *= r_lambda
    new_movie_mean_bias /= r_lambda * len(ratings) + r_gamma

    movie_var_bias[movie_idx] = 1.0 / (r_lambda * len(ratings) + r_gamma)
    movie_mean_bias[movie_idx] = new_movie_mean_bias

    # Embedding update
    users_mat = user_mean_embedding[users].T @ user_mean_embedding[users]
    users_mat += np.diag(user_var_embedding[users].sum(axis=0))
    users_mat *= r_lambda
    users_mat += r_tau * np.eye(embedding_dim)

    user_vec = user_mean_embedding[users].T @ (
        ratings - user_mean_bias[users] - movie_mean_bias[movie_idx]
    )
    user_vec *= r_lambda

    inv_user_mat = np.linalg.inv(users_mat)
    new_movie_mean_embedding = inv_user_mat @ user_vec
    new_movie_var_embedding = 1.0 / np.diag(users_mat)

    movie_var_embedding[movie_idx] = new_movie_var_embedding
    movie_mean_embedding[movie_idx] = new_movie_mean_embedding


@nb.njit(parallel=True)
def training_loop(
    train_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    val_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    embedding_dim: int = 2,
    r_lambda: float = 0.5,
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

    user_mean_bias = np.random.randn(n_user)
    movie_mean_bias = np.random.randn(n_movie)
    user_var_bias = np.random.randn(n_user)
    movie_var_bias = np.random.randn(n_movie)

    user_mean_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    user_var_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    movie_mean_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    )
    movie_var_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    )

    for iter in range(n_iter):
        for user_idx in nb.prange(len(train_user_movies)):
            update_users(
                user_idx,
                r_lambda,
                r_gamma,
                r_tau,
                train_user_movies,
                user_mean_embeddings,
                user_var_embeddings,
                user_mean_bias,
                user_var_bias,
                movie_mean_embeddings,
                movie_var_embeddings,
                movie_mean_bias,
            )

        for movie_idx in nb.prange(len(train_movie_users)):
            update_movie(
                movie_idx,
                r_lambda,
                r_gamma,
                r_tau,
                train_movie_users,
                user_mean_embeddings,
                user_var_embeddings,
                user_mean_bias,
                movie_mean_embeddings,
                movie_var_embeddings,
                movie_mean_bias,
                movie_var_bias,
            )

        train_loss[iter] = compute_elbo(
            r_lambda,
            r_gamma,
            r_tau,
            train_user_movies,
            user_mean_embeddings,
            user_var_embeddings,
            user_mean_bias,
            user_var_bias,
            movie_mean_embeddings,
            movie_var_embeddings,
            movie_mean_bias,
            movie_var_bias,
        )

        val_loss[iter] = compute_elbo(
            r_lambda,
            r_gamma,
            r_tau,
            val_user_movies,
            user_mean_embeddings,
            user_var_embeddings,
            user_mean_bias,
            user_var_bias,
            movie_mean_embeddings,
            movie_var_embeddings,
            movie_mean_bias,
            movie_var_bias,
        )

        train_rmse[iter] = compute_rmse(
            train_user_movies,
            user_mean_bias,
            movie_mean_bias,
            user_mean_embeddings,
            movie_mean_embeddings,
        )
        val_rmse[iter] = compute_rmse(
            val_user_movies,
            user_mean_bias,
            movie_mean_bias,
            user_mean_embeddings,
            movie_mean_embeddings,
        )

    return (
        train_loss,
        val_loss,
        train_rmse,
        val_rmse,
        user_mean_embeddings,
        user_var_embeddings,
        user_mean_bias,
        user_var_bias,
        movie_mean_embeddings,
        movie_var_embeddings,
        movie_mean_bias,
        movie_var_bias,
    )
