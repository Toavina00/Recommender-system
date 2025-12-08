from typing import Dict

import numpy as np

from ..utils.dataset import Dataset, Metadata
from .__base import training_loop as base_loop
from .__feature import training_loop as feat_loop
from .__variational import training_loop as var_loop


def train_base(
    train: Dataset,
    val: Dataset,
    n_iter: int = 10,
    embedding_dim: int = 2,
    r_lambda: float = 1.0,
    r_gamma: float = 0.1,
    r_tau: float = 0.1,
) -> Dict:
    (
        train_loss,
        train_rmse,
        val_loss,
        val_rmse,
        user_bias,
        movie_bias,
        user_embedding,
        movie_embedding,
    ) = base_loop(
        train.user_ptr,
        train.user_movies,
        train.user_ratings,
        train.movie_ptr,
        train.movie_users,
        train.movie_ratings,
        val.user_ptr,
        val.user_movies,
        val.user_ratings,
        val.movie_ptr,
        val.movie_users,
        val.movie_ratings,
        embedding_dim,
        r_lambda,
        r_gamma,
        r_tau,
        n_iter,
    )

    return {
        "train_loss": train_loss,
        "train_rmse": train_rmse,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "embeddings": {
            "user_bias": user_bias,
            "movie_bias": movie_bias,
            "user_embedding": user_embedding,
            "movie_embedding": movie_embedding,
        }
    }


def train_feat(
    train: Dataset,
    val: Dataset,
    metadata: Metadata,
    n_iter: int = 10,
    embedding_dim: int = 2,
    r_lambda: float = 1.0,
    r_gamma: float = 0.1,
    r_tau: float = 0.1,
) -> Dict:
    movie_feat = [np.array(arr) for arr in metadata.movie_feat]
    feat_movie = [np.array(arr) for arr in metadata.feat_movie]

    (
        train_loss,
        train_rmse,
        val_loss,
        val_rmse,
        user_bias,
        movie_bias,
        user_embedding,
        movie_embedding,
        feat_embedding,
    ) = feat_loop(
        train.user_ptr,
        train.user_movies,
        train.user_ratings,
        train.movie_ptr,
        train.movie_users,
        train.movie_ratings,
        val.user_ptr,
        val.user_movies,
        val.user_ratings,
        val.movie_ptr,
        val.movie_users,
        val.movie_ratings,
        movie_feat,
        feat_movie,
        embedding_dim,
        r_lambda,
        r_gamma,
        r_tau,
        n_iter,
    )

    return {
        "train_loss": train_loss,
        "train_rmse": train_rmse,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "embeddings": {
            "user_bias": user_bias,
            "movie_bias": movie_bias,
            "user_embedding": user_embedding,
            "movie_embedding": movie_embedding,
            "feat_embedding": feat_embedding,
        }
    }


def train_variational(
    train: Dataset,
    val: Dataset,
    n_iter: int = 10,
    embedding_dim: int = 2,
    r_lambda: float = 1.0,
    r_gamma: float = 0.1,
    r_tau: float = 0.1,
) -> Dict:
    (
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
    ) = var_loop(
        train.user_ptr,
        train.user_movies,
        train.user_ratings,
        train.movie_ptr,
        train.movie_users,
        train.movie_ratings,
        val.user_ptr,
        val.user_movies,
        val.user_ratings,
        embedding_dim,
        r_lambda,
        r_gamma,
        r_tau,
        n_iter,
    )

    return {
        "train_loss": train_loss,
        "train_rmse": train_rmse,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "embeddings": {
            "user_mean_embedding": user_mean_embedding,
            "user_var_embedding": user_var_embedding,
            "user_mean_bias": user_mean_bias,
            "user_var_bias": user_var_bias,
            "movie_mean_embedding": movie_mean_embedding,
            "movie_var_embedding": movie_var_embedding,
            "movie_mean_bias": movie_mean_bias,
            "movie_var_bias": movie_var_bias,
        }
    }
