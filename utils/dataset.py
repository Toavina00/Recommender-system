import csv
from typing import Any, Dict, List, Set, Tuple

import numpy as np


class Dataset:
    __slots__ = (
        "user_movies",
        "movie_users",
    )

    __movie_feat: List[Set[int] | np.ndarray] = list()
    __feat_movie: List[Set[int] | np.ndarray] = list()
    __idx_to_user_id: List[Any] = list()
    __idx_to_movie_id: List[Any] = list()
    __idx_to_feat_id: List[Any] = list()
    __user_id_to_idx: Dict[Any, int] = dict()
    __movie_id_to_idx: Dict[Any, int] = dict()
    __feat_id_to_idx: Dict[Any, int] = dict()

    @property
    def movie_feat(self):
        return type(self).__movie_feat

    @property
    def feat_movie(self):
        return type(self).__feat_movie

    @property
    def idx_to_user_id(self):
        return type(self).__idx_to_user_id

    @property
    def idx_to_movie_id(self):
        return type(self).__idx_to_movie_id

    @property
    def idx_to_feat_id(self):
        return type(self).__idx_to_feat_id

    @property
    def user_id_to_idx(self):
        return type(self).__user_id_to_idx

    @property
    def movie_id_to_idx(self):
        return type(self).__movie_id_to_idx

    @property
    def feat_id_to_idx(self):
        return type(self).__feat_id_to_idx

    def __init__(self):
        self.user_movies: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()
        self.movie_users: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()

    def __register_entry(self, user_id: Any, movie_id: Any, movie_feat: List[Any]):
        if user_id not in self.user_id_to_idx:
            self.user_id_to_idx[user_id] = len(self.idx_to_user_id)
            self.idx_to_user_id.append(user_id)

        if movie_id not in self.movie_id_to_idx:
            self.movie_id_to_idx[movie_id] = len(self.idx_to_movie_id)
            self.idx_to_movie_id.append(movie_id)
            self.movie_feat.append(set())

        for feat_id in movie_feat:
            if feat_id not in self.feat_id_to_idx:
                self.feat_id_to_idx[feat_id] = len(self.feat_id_to_idx)
                self.idx_to_feat_id.append(feat_id)
                self.feat_movie.append(set())

            f_idx = self.feat_id_to_idx[feat_id]
            m_idx = self.movie_id_to_idx[movie_id]
            self.movie_feat[m_idx].add(f_idx)  # pyright: ignore
            self.feat_movie[f_idx].add(m_idx)  # pyright: ignore

    def add_entry(
        self, user_id: Any, movie_id: Any, movie_feat: List[Any], rating: float
    ):
        self.__register_entry(user_id, movie_id, movie_feat)

        self.user_movies.extend(
            [([], []) for _ in range(len(self.user_movies), len(self.idx_to_user_id))]
        )
        self.movie_users.extend(
            [([], []) for _ in range(len(self.movie_users), len(self.idx_to_movie_id))]
        )

        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        self.user_movies[u_idx][0].append(m_idx)  # pyright: ignore
        self.user_movies[u_idx][1].append(rating)  # pyright: ignore
        self.movie_users[m_idx][0].append(u_idx)  # pyright: ignore
        self.movie_users[m_idx][1].append(rating)  # pyright: ignore

    def convert(self):
        self.user_movies.extend(
            [([], []) for _ in range(len(self.user_movies), len(self.idx_to_user_id))]
        )
        self.movie_users.extend(
            [([], []) for _ in range(len(self.movie_users), len(self.idx_to_movie_id))]
        )
        for i in range(len(self.user_movies)):
            self.user_movies[i] = (
                np.array(self.user_movies[i][0], dtype=int),
                np.array(self.user_movies[i][1], dtype=np.float64),
            )

        for i in range(len(self.movie_users)):
            self.movie_users[i] = (
                np.array(self.movie_users[i][0], dtype=int),
                np.array(self.movie_users[i][1], dtype=np.float64),
            )

        for i in range(len(self.movie_feat)):
            if isinstance(self.movie_feat[i], set):
                self.movie_feat[i] = np.array(list(self.movie_feat[i]), dtype=int)

        for i in range(len(self.feat_movie)):
            if isinstance(self.feat_movie[i], set):
                self.feat_movie[i] = np.array(list(self.feat_movie[i]), dtype=int)

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
                test.add_entry(user_id, movie_id, features, float(rating))
            else:
                train.add_entry(user_id, movie_id, features, float(rating))

    train.convert()
    test.convert()

    return (train, test)
