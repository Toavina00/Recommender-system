from typing import List, Tuple

import numba as nb
import numpy as np


@nb.njit
def compute_loss(
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    movie_ptr: np.ndarray,
    movie_users: np.ndarray,
    movie_ratings: np.ndarray,
    movie_feat: List[np.ndarray],
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embedding: np.ndarray,
    movie_embedding: np.ndarray,
    feat_embedding: np.ndarray,
) -> float:
    loss = 0.0
    shifed_movie_embedding = np.zeros_like(movie_embedding)

    for movie_idx in range(len(movie_ptr) - 1):
        users = movie_users[movie_ptr[movie_idx] : movie_ptr[movie_idx + 1]]
        ratings = movie_ratings[movie_ptr[movie_idx] : movie_ptr[movie_idx + 1]]

        if len(ratings) == 0:
            continue

        e_vec = ratings - (
            user_embedding[users] @ movie_embedding[movie_idx]
            + user_bias[users]
            + movie_bias[movie_idx]
        )
        e_sum = e_vec @ e_vec
        loss += e_sum

        feat_movie = feat_embedding[movie_feat[movie_idx]].sum(axis=0)
        feat_movie /= np.sqrt(len(movie_feat[movie_idx]))
        shifed_movie_embedding[movie_idx] = movie_embedding[movie_idx] - feat_movie

    loss *= r_lambda * 0.5
    loss += r_gamma * 0.5 * (user_bias @ user_bias)
    loss += r_gamma * 0.5 * (movie_bias @ movie_bias)
    loss += r_tau * 0.5 * np.sum(feat_embedding * feat_embedding)
    loss += r_tau * 0.5 * np.sum(user_embedding * user_embedding)
    loss += r_tau * 0.5 * np.sum(shifed_movie_embedding * shifed_movie_embedding)

    return loss


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
def update_user_bias(
    user_idx: int,
    r_lambda: float,
    r_gamma: float,
    movies: np.ndarray,
    ratings: np.ndarray,
    user_embedding: np.ndarray,
    user_bias: np.ndarray,
    movie_embedding: np.ndarray,
    movie_bias: np.ndarray,
):
    if len(ratings) == 0:
        return

    new_user_bias = np.sum(
        ratings
        - (movie_embedding[movies] @ user_embedding[user_idx] + movie_bias[movies])
    )
    new_user_bias *= r_lambda
    new_user_bias /= r_lambda * len(ratings) + r_gamma

    user_bias[user_idx] = new_user_bias


@nb.njit
def update_user_embedding(
    user_idx: int,
    r_lambda: float,
    r_tau: float,
    movies: np.ndarray,
    ratings: np.ndarray,
    user_embedding: np.ndarray,
    user_bias: np.ndarray,
    movie_embedding: np.ndarray,
    movie_bias: np.ndarray,
):
    embedding_dim = user_embedding.shape[1]

    if len(ratings) == 0:
        return

    movie_mat = movie_embedding[movies].T @ movie_embedding[movies]
    movie_vec = movie_embedding[movies].T @ (
        ratings - movie_bias[movies] - user_bias[user_idx]
    )

    movie_mat *= r_lambda
    movie_mat += r_tau * np.eye(embedding_dim)
    movie_vec *= r_lambda

    inv_movie_mat = np.linalg.inv(movie_mat)
    new_user_embedding = inv_movie_mat @ movie_vec

    user_embedding[user_idx] = new_user_embedding


@nb.njit
def update_movie_bias(
    movie_idx: int,
    r_lambda: float,
    r_gamma: float,
    users: np.ndarray,
    ratings: np.ndarray,
    user_embedding: np.ndarray,
    user_bias: np.ndarray,
    movie_embedding: np.ndarray,
    movie_bias: np.ndarray,
):
    if len(ratings) == 0:
        return

    new_movie_bias = np.sum(
        ratings
        - (user_embedding[users] @ movie_embedding[movie_idx] + user_bias[users])
    )
    new_movie_bias *= r_lambda
    new_movie_bias /= r_lambda * len(ratings) + r_gamma

    movie_bias[movie_idx] = new_movie_bias


@nb.njit
def update_movie_embedding(
    movie_idx: int,
    r_lambda: float,
    r_tau: float,
    users: np.ndarray,
    ratings: np.ndarray,
    movie_feat: List[np.ndarray],
    user_embedding: np.ndarray,
    user_bias: np.ndarray,
    movie_embedding: np.ndarray,
    movie_bias: np.ndarray,
    feat_embedding: np.ndarray,
):
    embedding_dim = movie_embedding.shape[1]

    if len(ratings) == 0:
        return

    users_mat = user_embedding[users].T @ user_embedding[users]
    feature_vec = feat_embedding[movie_feat[movie_idx]].sum(axis=0) / np.sqrt(
        len(movie_feat[movie_idx])
    )
    user_vec = user_embedding[users].T @ (
        ratings - user_bias[users] - movie_bias[movie_idx]
    )

    users_mat *= r_lambda
    users_mat += r_tau * np.eye(embedding_dim)
    user_vec *= r_lambda
    user_vec += r_tau * feature_vec

    inv_user_mat = np.linalg.inv(users_mat)
    new_movie_embedding = inv_user_mat @ user_vec

    movie_embedding[movie_idx] = new_movie_embedding


@nb.njit
def update_feature_embedding(
    feat_idx: int,
    movie_feat: List[np.ndarray],
    feat_movie: List[np.ndarray],
    movie_embedding: np.ndarray,
    feat_embedding: np.ndarray,
):
    embedding_dim = feat_embedding.shape[1]

    if len(feat_movie[feat_idx]) == 0:
        return

    scale = np.zeros((len(feat_movie[feat_idx])))
    acc_f = np.zeros((len(feat_movie[feat_idx]), embedding_dim))
    for i, movie_idx in enumerate(feat_movie[feat_idx]):
        scale[i] = 1.0 / np.sqrt(len(movie_feat[movie_idx]))
        acc_f[i] = feat_embedding[movie_feat[movie_idx]].sum(axis=0)
        acc_f[i] -= feat_embedding[feat_idx]
        acc_f[i] *= scale[i]

    acc_f = movie_embedding[feat_movie[feat_idx]] - acc_f

    for i in range(len(feat_movie[feat_idx])):
        acc_f[i] *= scale[i]

    acc_f = acc_f.sum(axis=0)
    scale = (scale**2).sum() + 1.0

    feat_embedding[feat_idx] = acc_f / scale


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
    val_movie_ptr: np.ndarray,
    val_movie_users: np.ndarray,
    val_movie_ratings: np.ndarray,
    movie_feat: List[np.ndarray],
    feat_movie: List[np.ndarray],
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
]:
    train_loss = np.zeros((n_iter))
    train_rmse = np.zeros((n_iter))
    val_loss = np.zeros((n_iter))
    val_rmse = np.zeros((n_iter))

    n_user, n_movie, n_feat = (
        len(train_user_ptr) - 1,
        len(train_movie_ptr) - 1,
        len(feat_movie),
    )

    user_bias = np.random.randn(n_user)
    movie_bias = np.random.randn(n_movie)
    user_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    movie_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    )
    feat_embedding = np.random.normal(
        0, np.sqrt(embedding_dim), (n_feat, embedding_dim)
    )

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
                user_embedding,
                user_bias,
                movie_embedding,
                movie_bias,
            )

            update_user_embedding(
                user_idx,
                r_lambda,
                r_tau,
                movies,
                ratings,
                user_embedding,
                user_bias,
                movie_embedding,
                movie_bias,
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
                user_embedding,
                user_bias,
                movie_embedding,
                movie_bias,
            )

            update_movie_embedding(
                movie_idx,
                r_lambda,
                r_tau,
                users,
                ratings,
                movie_feat,
                user_embedding,
                user_bias,
                movie_embedding,
                movie_bias,
                feat_embedding,
            )

        new_feat_embedding = feat_embedding.copy()

        for feat_idx in nb.prange(n_feat):
            update_feature_embedding(
                feat_idx,
                movie_feat,
                feat_movie,
                movie_embedding,
                new_feat_embedding,
            )

        feat_embedding = new_feat_embedding

        train_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            train_movie_ptr,
            train_movie_users,
            train_movie_ratings,
            movie_feat,
            user_bias,
            movie_bias,
            user_embedding,
            movie_embedding,
            feat_embedding,
        )

        train_rmse[iter] = compute_rmse(
            train_user_ptr,
            train_user_movies,
            train_user_ratings,
            user_bias,
            movie_bias,
            user_embedding,
            movie_embedding,
        )

        val_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            val_movie_ptr,
            val_movie_users,
            val_movie_ratings,
            movie_feat,
            user_bias,
            movie_bias,
            user_embedding,
            movie_embedding,
            feat_embedding,
        )

        val_rmse[iter] = compute_rmse(
            val_user_ptr,
            val_user_movies,
            val_user_ratings,
            user_bias,
            movie_bias,
            user_embedding,
            movie_embedding,
        )

    return (
        train_loss,
        train_rmse,
        val_loss,
        val_rmse,
        user_bias,
        movie_bias,
        user_embedding,
        movie_embedding,
        feat_embedding,
    )
