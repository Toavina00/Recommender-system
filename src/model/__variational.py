from typing import Tuple

import numba as nb
import numpy as np


@nb.njit
def compute_rmse(
    user_ptr: np.ndarray,
    user_movies: np.ndarray,
    user_ratings: np.ndarray,
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embedding: np.ndarray,
    movie_embedding: np.ndarray,
) -> float:
    error = 0.0

    for user_idx in range(len(user_ptr) - 1):
        movies = user_movies[user_ptr[user_idx] : user_ptr[user_idx + 1]]
        ratings = user_ratings[user_ptr[user_idx] : user_ptr[user_idx + 1]]

        if len(ratings) == 0:
            continue

        e_vec = ratings - (
            movie_embedding[movies] @ user_embedding[user_idx]
            + movie_bias[movies]
            + user_bias[user_idx]
        )
        e_sum = e_vec @ e_vec
        error += e_sum

    error /= len(user_ratings)
    error = np.sqrt(error)

    return error


@nb.njit
def compute_elbo(
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_ptr: np.ndarray,
    user_movies: np.ndarray,
    user_ratings: np.ndarray,
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    user_var_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
    movie_var_bias: np.ndarray,
) -> float:
    elbo = 0.0

    for user_idx in range(len(user_ptr) - 1):
        movies = user_movies[user_ptr[user_idx] : user_ptr[user_idx + 1]]
        ratings = user_ratings[user_ptr[user_idx] : user_ptr[user_idx + 1]]

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

        elbo += (user_mean_bias[user_idx] ** 2 + user_var_bias[user_idx]) * len(movies)
        elbo += np.sum(movie_mean_bias[movies] ** 2 + movie_var_bias[movies])

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
    elbo -= 0.5 * r_gamma * np.sum(user_mean_bias**2 + user_var_bias)
    elbo -= 0.5 * r_gamma * np.sum(movie_mean_bias**2 + movie_var_bias)
    elbo += 0.5 * np.sum(np.log(user_var_embedding))
    elbo += 0.5 * np.sum(np.log(movie_var_embedding))
    elbo += 0.5 * np.sum(np.log(user_var_bias))
    elbo += 0.5 * np.sum(np.log(movie_var_bias))

    return elbo


@nb.njit
def update_user_bias(
    user_idx: int,
    r_lambda: float,
    r_gamma: float,
    movies: np.ndarray,
    ratings: np.ndarray,
    user_mean_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    user_var_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
):
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
    new_user_var_bias = 1.0 / (r_lambda * len(ratings) + r_gamma)
    new_user_mean_bias *= new_user_var_bias

    user_var_bias[user_idx] = new_user_var_bias
    user_mean_bias[user_idx] = new_user_mean_bias


@nb.njit
def update_user_embedding(
    user_idx: int,
    r_lambda: float,
    r_tau: float,
    movies: np.ndarray,
    ratings: np.ndarray,
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
):
    embedding_dim = user_mean_embedding.shape[1]

    if len(ratings) == 0:
        return

    # Embedding update
    movie_mat = movie_mean_embedding[movies].T @ movie_mean_embedding[movies]
    movie_mat += np.diag(movie_var_embedding[movies].sum(axis=0))
    movie_mat *= r_lambda
    movie_mat += r_tau * np.eye(embedding_dim)

    movie_vec = movie_mean_embedding[movies].T @ (
        ratings - movie_mean_bias[movies] - user_mean_bias[user_idx]
    )
    movie_vec *= r_lambda

    inv_movie_mat = np.linalg.inv(movie_mat)
    new_user_mean_embedding = inv_movie_mat @ movie_vec
    new_user_var_embedding = 1.0 / np.diag(movie_mat)

    user_var_embedding[user_idx] = new_user_var_embedding
    user_mean_embedding[user_idx] = new_user_mean_embedding


@nb.njit
def update_movie_bias(
    movie_idx: int,
    r_lambda: float,
    r_gamma: float,
    users: np.ndarray,
    ratings: np.ndarray,
    user_mean_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
    movie_var_bias: np.ndarray,
):
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
    new_movie_var_bias = 1.0 / (r_lambda * len(ratings) + r_gamma)
    new_movie_mean_bias *= new_movie_var_bias

    movie_var_bias[movie_idx] = new_movie_var_bias
    movie_mean_bias[movie_idx] = new_movie_mean_bias


@nb.njit
def update_movie_embedding(
    movie_idx: int,
    r_lambda: float,
    r_tau: float,
    users: np.ndarray,
    ratings: np.ndarray,
    user_mean_embedding: np.ndarray,
    user_var_embedding: np.ndarray,
    user_mean_bias: np.ndarray,
    movie_mean_embedding: np.ndarray,
    movie_var_embedding: np.ndarray,
    movie_mean_bias: np.ndarray,
):
    embedding_dim = movie_mean_embedding.shape[1]

    if len(ratings) == 0:
        return

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
    train_user_ptr: np.ndarray,
    train_user_movies: np.ndarray,
    train_user_ratings: np.ndarray,
    train_movie_ptr: np.ndarray,
    train_movie_users: np.ndarray,
    train_movie_ratings: np.ndarray,
    val_user_ptr: np.ndarray,
    val_user_movies: np.ndarray,
    val_user_ratings: np.ndarray,
    embedding_dim: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    n_iter: int,
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
        len(train_user_ptr) - 1,
        len(train_movie_ptr) - 1,
    )

    user_mean_bias = np.random.randn(n_user)
    movie_mean_bias = np.random.randn(n_movie)
    user_var_bias = np.random.chisquare(1, (n_user,))
    movie_var_bias = np.random.chisquare(1, (n_movie,))

    user_mean_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    user_var_embedding = np.random.chisquare(embedding_dim, (n_user, embedding_dim))
    movie_mean_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    )
    movie_var_embedding = np.random.chisquare(embedding_dim, (n_movie, embedding_dim))

    for iter in range(n_iter):
        for user_idx in nb.prange(n_user):
            movies = train_user_movies[
                train_user_ptr[user_idx] : train_user_ptr[user_idx + 1]
            ]
            ratings = train_user_ratings[
                train_user_ptr[user_idx] : train_user_ptr[user_idx + 1]
            ]
            update_user_bias(
                user_idx,
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

        for user_idx in nb.prange(n_user):
            movies = train_user_movies[
                train_user_ptr[user_idx] : train_user_ptr[user_idx + 1]
            ]
            ratings = train_user_ratings[
                train_user_ptr[user_idx] : train_user_ptr[user_idx + 1]
            ]
            update_user_embedding(
                user_idx,
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

        for movie_idx in nb.prange(n_movie):
            users = train_movie_users[
                train_movie_ptr[movie_idx] : train_movie_ptr[movie_idx + 1]
            ]
            ratings = train_movie_ratings[
                train_movie_ptr[movie_idx] : train_movie_ptr[movie_idx + 1]
            ]
            update_movie_bias(
                movie_idx,
                r_lambda,
                r_gamma,
                users,
                ratings,
                user_mean_embedding,
                user_mean_bias,
                movie_mean_embedding,
                movie_mean_bias,
                movie_var_bias,
            )

        for movie_idx in nb.prange(n_movie):
            users = train_movie_users[
                train_movie_ptr[movie_idx] : train_movie_ptr[movie_idx + 1]
            ]
            ratings = train_movie_ratings[
                train_movie_ptr[movie_idx] : train_movie_ptr[movie_idx + 1]
            ]
            update_movie_embedding(
                movie_idx,
                r_lambda,
                r_tau,
                users,
                ratings,
                user_mean_embedding,
                user_var_embedding,
                user_mean_bias,
                movie_mean_embedding,
                movie_var_embedding,
                movie_mean_bias,
            )

        train_loss[iter] = compute_elbo(
            r_lambda,
            r_gamma,
            r_tau,
            train_user_ptr,
            train_user_movies,
            train_user_ratings,
            user_mean_embedding,
            user_var_embedding,
            user_mean_bias,
            user_var_bias,
            movie_mean_embedding,
            movie_var_embedding,
            movie_mean_bias,
            movie_var_bias,
        )

        val_loss[iter] = compute_elbo(
            r_lambda,
            r_gamma,
            r_tau,
            val_user_ptr,
            val_user_movies,
            val_user_ratings,
            user_mean_embedding,
            user_var_embedding,
            user_mean_bias,
            user_var_bias,
            movie_mean_embedding,
            movie_var_embedding,
            movie_mean_bias,
            movie_var_bias,
        )

        train_rmse[iter] = compute_rmse(
            train_user_ptr,
            train_user_movies,
            train_user_ratings,
            user_mean_bias,
            movie_mean_bias,
            user_mean_embedding,
            movie_mean_embedding,
        )

        val_rmse[iter] = compute_rmse(
            val_user_ptr,
            val_user_movies,
            val_user_ratings,
            user_mean_bias,
            movie_mean_bias,
            user_mean_embedding,
            movie_mean_embedding,
        )

    return (
        train_loss,
        train_rmse,
        val_loss,
        val_rmse,
        user_mean_embedding,
        user_var_embedding,
        user_mean_bias,
        user_var_bias,
        movie_mean_embedding,
        movie_var_embedding,
        movie_mean_bias,
        movie_var_bias,
    )
