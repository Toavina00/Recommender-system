import csv
from typing import Any, Dict, List, Tuple

import numpy as np


class Dataset:
    __slots__ = (
        "user_movies",
        "movie_users",
        "movie_feat",
        "feat_movie",
        "idx_to_user_id",
        "idx_to_movie_id",
        "idx_to_feat_id",
        "user_id_to_idx",
        "movie_id_to_idx",
        "feat_id_to_idx",
    )

    def __init__(self):
        self.user_movies: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()
        self.movie_users: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()

        self.movie_feat: List[List[int] | np.ndarray] = list()
        self.feat_movie: List[List[int] | np.ndarray] = list()

        self.idx_to_user_id: List[Any] = list()
        self.idx_to_movie_id: List[Any] = list()
        self.idx_to_feat_id: List[Any] = list()

        self.user_id_to_idx: Dict[Any, int] = dict()
        self.movie_id_to_idx: Dict[Any, int] = dict()
        self.feat_id_to_idx: Dict[Any, int] = dict()

    def register_entry(self, user_id: Any, movie_id: Any, movie_feat: List[Any]):
        if user_id not in self.user_id_to_idx:
            self.user_id_to_idx[user_id] = len(self.idx_to_user_id)
            self.idx_to_user_id.append(user_id)
            self.user_movies.append(([], []))

        if movie_id not in self.movie_id_to_idx:
            self.movie_id_to_idx[movie_id] = len(self.idx_to_movie_id)
            self.idx_to_movie_id.append(movie_id)
            self.movie_users.append(([], []))
            self.movie_feat.append([])

        for feat_id in movie_feat:
            if feat_id not in self.feat_id_to_idx:
                self.feat_id_to_idx[feat_id] = len(self.feat_id_to_idx)
                self.idx_to_feat_id.append(feat_id)
                self.feat_movie.append([])

            f_idx = self.feat_id_to_idx[feat_id]
            m_idx = self.movie_id_to_idx[movie_id]
            self.feat_movie[f_idx].append(m_idx)  # pyright: ignore
            self.movie_feat[m_idx].append(f_idx)  # pyright: ignore

    def add_entry(
        self, user_id: Any, movie_id: Any, movie_feat: List[Any], rating: float
    ):
        self.register_entry(user_id, movie_id, movie_feat)

        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]

        self.user_movies[u_idx][0].append(m_idx)  # pyright: ignore
        self.user_movies[u_idx][1].append(rating)  # pyright: ignore
        self.movie_users[m_idx][0].append(u_idx)  # pyright: ignore
        self.movie_users[m_idx][1].append(rating)  # pyright: ignore

    def convert(self):
        for i in range(len(self.user_movies)):
            self.user_movies[i] = (
                np.array(self.user_movies[i][0], dtype=int),
                np.array(self.user_movies[i][1], dtype=float),
            )
        for i in range(len(self.movie_users)):
            self.movie_users[i] = (
                np.array(self.movie_users[i][0], dtype=int),
                np.array(self.movie_users[i][1], dtype=float),
            )
        for i in range(len(self.movie_feat)):
            self.movie_feat[i] = np.array(self.movie_feat[i], dtype=int)
        for i in range(len(self.feat_movie)):
            self.feat_movie[i] = np.array(self.feat_movie[i], dtype=int)

    def __len__(self) -> int:
        length = 0
        for i in range(len(self.user_movies)):
            length += len(self.user_movies[i][0])

        return length


def train_test_load(
    rating_path: str, movie_path: str, test_split: float = 0.2
) -> Tuple[Dataset, Dataset]:
    movie_id_feat = dict()

    with open(movie_path, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for line in reader:
            movie_id, features = line[0], line[-1]
            movie_id_feat[movie_id] = features

    train, test = Dataset(), Dataset()

    with open(rating_path, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for line in reader:
            user_id, movie_id, rating, _ = line
            features = movie_id_feat[movie_id].split("|")

            if np.random.rand() < test_split:
                train.register_entry(user_id, movie_id, features)
                test.add_entry(user_id, movie_id, features, float(rating))
            else:
                train.add_entry(user_id, movie_id, features, float(rating))
                test.register_entry(user_id, movie_id, features)

    train.convert()
    test.convert()

    return (train, test)
