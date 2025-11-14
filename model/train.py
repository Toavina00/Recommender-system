from typing import List, Tuple

import numba as nb
import numpy as np


#@nb.njit(cache=True)
def compute_loss(
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    movie_users: List[Tuple[np.ndarray, np.ndarray]],
    movie_feat: List[np.ndarray],
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    feat_embeddings: np.ndarray,
) -> float:
    loss = 0.0
    shifed_movie_embeddings = np.zeros_like(movie_embeddings)

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

        feat_movie = feat_embeddings[movie_feat[movie_idx]].sum(axis=0)
        feat_movie /= np.sqrt(len(movie_feat[movie_idx]))
        shifed_movie_embeddings[movie_idx] = movie_embeddings[movie_idx] - feat_movie

    loss *= r_lambda * 0.5
    loss += r_gamma * 0.5 * (user_bias @ user_bias)
    loss += r_gamma * 0.5 * (movie_bias @ movie_bias)
    loss += r_tau * 0.5 * np.sum(user_embeddings * user_embeddings)
    loss += r_tau * 0.5 * np.sum(shifed_movie_embeddings * shifed_movie_embeddings)

    return loss


#@nb.njit(cache=True)
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


#@nb.njit(parallel=True, cache=True)
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

        new_user_bias = np.sum(
            ratings
            - (
                movie_embeddings[movies] @ user_embeddings[user_idx]
                + movie_bias[movies]
            )
        )
        movie_mat = movie_embeddings[movies].T @ movie_embeddings[movies]
        movie_vec = movie_embeddings[movies].T @ (
            ratings - movie_bias[movies] - user_bias[user_idx]
        )

        movie_mat *= r_lambda
        movie_mat += r_tau * np.eye(embedding_dim)
        movie_vec *= r_lambda

        new_user_bias *= r_lambda
        new_user_bias /= r_lambda * len(ratings) + r_gamma

        inv_movie_mat = np.linalg.inv(movie_mat)
        new_user_embedding = inv_movie_mat @ movie_vec

        user_embeddings[user_idx] = new_user_embedding
        user_bias[user_idx] = new_user_bias


#@nb.njit(parallel=True, cache=True)
def optimize_movie(
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    movie_feat: List[np.ndarray],
    embedding_dim: int,
    r_lambda: float,
    r_gamma: float,
    r_tau: float,
    user_bias: np.ndarray,
    movie_bias: np.ndarray,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    feat_embeddings: np.ndarray,
) -> None:
    for movie_idx in nb.prange(len(train_movie_users)):
        users, ratings = train_movie_users[movie_idx]

        if len(ratings) == 0:
            continue

        new_movie_bias = np.sum(
            ratings
            - (user_embeddings[users] @ movie_embeddings[movie_idx] + user_bias[users])
        )
        users_mat = user_embeddings[users].T @ user_embeddings[users]
        feature_vec = feat_embeddings[movie_feat[movie_idx]].sum(axis=0) / np.sqrt(
            len(movie_feat[movie_idx])
        )
        user_vec = user_embeddings[users].T @ (
            ratings - user_bias[users] - movie_bias[movie_idx]
        ) 

        users_mat *= r_lambda
        users_mat += r_tau * np.eye(embedding_dim)
        user_vec *= r_lambda
        user_vec += r_tau * feature_vec

        new_movie_bias *= r_lambda
        new_movie_bias /= r_lambda * len(ratings) + r_gamma

        inv_user_mat = np.linalg.inv(users_mat)
        new_movie_embedding = inv_user_mat @ user_vec

        movie_embeddings[movie_idx] = new_movie_embedding
        movie_bias[movie_idx] = new_movie_bias


#@nb.njit(parallel=True, cache=True)
def optimize_features(
    movie_feat: List[np.ndarray],
    feat_movie: List[np.ndarray],
    embedding_dim: int,
    movie_embeddings: np.ndarray,
    feat_embeddings: np.ndarray,
):
    new_feat_embeddings = feat_embeddings.copy()

    for feat_idx in nb.prange(len(feat_embeddings)):
        if len(feat_movie[feat_idx]) == 0:
            continue

        scale = np.zeros((len(feat_movie[feat_idx])))
        acc_f = np.zeros((len(feat_movie[feat_idx]), embedding_dim))
        for i, movie_idx in enumerate(feat_movie[feat_idx]):
            scale[i] = 1.0 / np.sqrt(len(movie_feat[movie_idx]))
            acc_f[i] = feat_embeddings[movie_feat[movie_idx]].sum(axis=0)
            acc_f[i] -= feat_embeddings[feat_idx]

        acc_f = movie_embeddings[feat_movie[feat_idx]] - acc_f
        acc_f *= scale[:, np.newaxis]
        acc_f = acc_f.sum(axis=0)
        scale = scale.sum() + 1

        new_feat_embeddings[feat_idx] = acc_f / scale

    feat_embeddings = new_feat_embeddings


#@nb.njit(cache=True)
def training_loop(
    train_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    train_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    val_user_movies: List[Tuple[np.ndarray, np.ndarray]],
    val_movie_users: List[Tuple[np.ndarray, np.ndarray]],
    movie_feat: List[np.ndarray],
    feat_movie: List[np.ndarray],
    embedding_dim: int = 2,
    r_lambda: float = 0.05,
    r_gamma: float = 0.05,
    r_tau: float = 0.05,
    n_iter: int = 100,
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
        len(train_user_movies),
        len(train_movie_users),
        len(feat_movie),
    )

    user_bias = np.random.randn(n_user)
    movie_bias = np.random.randn(n_movie)
    user_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_user, embedding_dim)
    )
    movie_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_movie, embedding_dim)
    ) 
    feat_embeddings = np.random.normal(
        0, np.sqrt(embedding_dim), (n_feat, embedding_dim)
    )

    for iter in range(n_iter):
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
            movie_feat,
            embedding_dim,
            r_lambda,
            r_gamma,
            r_tau,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
            feat_embeddings,
        )
        optimize_features(
           movie_feat, feat_movie, embedding_dim, movie_embeddings, feat_embeddings
        )

        train_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            train_movie_users,
            movie_feat,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
            feat_embeddings,
        )
        train_rmse[iter] = compute_rmse(
            train_user_movies, user_bias, movie_bias, user_embeddings, movie_embeddings
        )
        val_loss[iter] = compute_loss(
            r_lambda,
            r_gamma,
            r_tau,
            val_movie_users,
            movie_feat,
            user_bias,
            movie_bias,
            user_embeddings,
            movie_embeddings,
            feat_embeddings,
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
        feat_embeddings,
    )
