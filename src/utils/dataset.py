import csv
from typing import Any, Dict, List, Tuple

import numpy as np


class Metadata:
    __slots__ = (
        "movie_feat",
        "feat_movie",
        "idx_to_user_id",
        "idx_to_movie_id",
        "idx_to_feat_id",
        "user_id_to_idx",
        "movie_id_to_idx",
        "feat_id_to_idx",
        "movie_title_to_id",
        "movie_id_to_title",
    )

    def __init__(self):
        self.movie_feat: List[List[int]] = list()
        self.feat_movie: List[List[int]] = list()
        self.idx_to_user_id: List[Any] = list()
        self.idx_to_movie_id: List[Any] = list()
        self.idx_to_feat_id: List[Any] = list()
        self.user_id_to_idx: Dict[Any, int] = dict()
        self.movie_id_to_idx: Dict[Any, int] = dict()
        self.feat_id_to_idx: Dict[Any, int] = dict()
        self.movie_title_to_id: Dict[Any, Any] = dict()
        self.movie_id_to_title: Dict[Any, Any] = dict()

    def register_user(self, user_id: Any):
        if user_id not in self.user_id_to_idx:
            self.user_id_to_idx[user_id] = len(self.idx_to_user_id)
            self.idx_to_user_id.append(user_id)

    def register_movie(self, movie_id: Any):
        if movie_id not in self.movie_id_to_idx:
            self.movie_id_to_idx[movie_id] = len(self.idx_to_movie_id)
            self.idx_to_movie_id.append(movie_id)

    def register_movie_feat(self, movie_id: Any, movie_feat: List[Any]):
        if movie_id in self.movie_id_to_idx:
            return

        self.movie_id_to_idx[movie_id] = len(self.idx_to_movie_id)
        self.idx_to_movie_id.append(movie_id)
        self.movie_feat.append(list())

        for feat_id in movie_feat:
            if feat_id not in self.feat_id_to_idx:
                self.feat_id_to_idx[feat_id] = len(self.feat_id_to_idx)
                self.idx_to_feat_id.append(feat_id)
                self.feat_movie.append(list())

            f_idx = self.feat_id_to_idx[feat_id]
            m_idx = self.movie_id_to_idx[movie_id]

            self.movie_feat[m_idx].append(f_idx)
            self.feat_movie[f_idx].append(m_idx)

    def register_movie_title(self, movie_id: Any, movie_title: Any):
        if movie_id in self.movie_id_to_title:
            return

        self.movie_id_to_title[movie_id] = movie_title
        self.movie_title_to_id[movie_title] = movie_id


class Dataset:
    __slots__ = (
        "user_ptr",
        "user_movies",
        "user_ratings",
        "movie_ptr",
        "movie_users",
        "movie_ratings",
    )

    def __init__(
        self,
        users: List[int],
        movies: List[int],
        ratings: List[float],
        metadata: Metadata,
    ):
        self.user_ptr: np.ndarray = np.concatenate(
            ([0], np.bincount(users, minlength=len(metadata.idx_to_user_id)).cumsum())
        )
        self.user_movies: np.ndarray = np.array(movies, dtype=np.int32)
        self.user_ratings: np.ndarray = np.array(ratings, dtype=np.float64)
        self.movie_ptr: np.ndarray = np.concatenate(
            ([0], np.bincount(movies, minlength=len(metadata.idx_to_movie_id)).cumsum())
        )
        self.movie_users: np.ndarray = np.array(users, dtype=np.int32)
        self.movie_ratings: np.ndarray = np.array(ratings, dtype=np.float64)

        user_order = np.argsort(self.movie_users)
        movie_order = np.argsort(self.user_movies)

        self.user_movies = self.user_movies[user_order]
        self.user_ratings = self.user_ratings[user_order]
        self.movie_users = self.movie_users[movie_order]
        self.movie_ratings = self.movie_ratings[movie_order]

    def __len__(self) -> int:
        return len(self.user_ratings)


def load_data(
    rating_path: str,
    movie_path: str,
    include_features: bool = False,
    include_title: bool = False,
) -> Tuple[Dataset, Metadata]:
    metadata = Metadata()

    if include_title or include_features:
        with open(movie_path, newline="") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for line in reader:
                movie_id, title, features = line
                if include_title:
                    metadata.register_movie_title(movie_id, title)
                if include_features:
                    metadata.register_movie_feat(movie_id, features.split("|"))

    users = []
    movies = []
    ratings = []

    with open(rating_path, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for line in reader:
            user_id, movie_id, rating, _ = line

            metadata.register_user(user_id)
            metadata.register_movie(movie_id)

            users.append(metadata.user_id_to_idx[user_id])
            movies.append(metadata.movie_id_to_idx[movie_id])
            ratings.append(float(rating))

    return (
        Dataset(users, movies, ratings, metadata),
        metadata,
    )


def train_test_load(
    rating_path: str,
    movie_path: str,
    test_split: float = 0.2,
    include_features: bool = False,
    include_title: bool = False,
) -> Tuple[Dataset, Dataset, Metadata]:
    metadata = Metadata()

    if include_title or include_features:
        with open(movie_path, newline="") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for line in reader:
                movie_id, title, features = line
                if include_title:
                    metadata.register_movie_title(movie_id, title)
                if include_features:
                    metadata.register_movie_feat(movie_id, features.split("|"))

    train_users = []
    train_movies = []
    train_ratings = []

    test_users = []
    test_movies = []
    test_ratings = []

    with open(rating_path, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for line in reader:
            user_id, movie_id, rating, _ = line

            metadata.register_user(user_id)
            metadata.register_movie(movie_id)

            if np.random.rand() < test_split:
                test_users.append(metadata.user_id_to_idx[user_id])
                test_movies.append(metadata.movie_id_to_idx[movie_id])
                test_ratings.append(float(rating))
            else:
                train_users.append(metadata.user_id_to_idx[user_id])
                train_movies.append(metadata.movie_id_to_idx[movie_id])
                train_ratings.append(float(rating))

    return (
        Dataset(train_users, train_movies, train_ratings, metadata),
        Dataset(test_users, test_movies, test_ratings, metadata),
        metadata,
    )
