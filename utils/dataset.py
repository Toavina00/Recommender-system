from typing import Any, Dict, List, Tuple

import numpy as np


class Dataset:
    __slots__ = (
        "user_movies",
        "movie_users",
        "idx_to_user_id",
        "idx_to_movie_id",
        "user_id_to_idx",
        "movie_id_to_idx",
    )

    def __init__(self):
        self.user_movies: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()
        self.movie_users: List[
            Tuple[List[int] | np.ndarray, List[float] | np.ndarray]
        ] = list()

        self.idx_to_user_id: List[Any] = list()
        self.idx_to_movie_id: List[Any] = list()

        self.user_id_to_idx: Dict[Any, int] = {}
        self.movie_id_to_idx: Dict[Any, int] = {}

    def register_entry(self, user_id: Any, movie_id: Any):
        if user_id not in self.user_id_to_idx:
            self.user_id_to_idx[user_id] = len(self.idx_to_user_id)
            self.idx_to_user_id.append(user_id)
            self.user_movies.append(([], []))

        if movie_id not in self.movie_id_to_idx:
            self.movie_id_to_idx[movie_id] = len(self.idx_to_movie_id)
            self.idx_to_movie_id.append(movie_id)
            self.movie_users.append(([], []))

    def add_entry(self, user_id: Any, movie_id: Any, rating: float):
        self.register_entry(user_id, movie_id)

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
                np.array(self.user_movies[i][1]),
            )
        for i in range(len(self.movie_users)):
            self.movie_users[i] = (
                np.array(self.movie_users[i][0], dtype=int),
                np.array(self.movie_users[i][1]),
            )

    def __len__(self) -> int:
        length = 0
        for i in range(len(self.user_movies)):
            length += len(self.user_movies[i][0])

        return length


def train_test_load(file_path: str, test_split: float = 0.2) -> Tuple[Dataset, Dataset]:
    train, test = Dataset(), Dataset()

    with open(file_path) as f:
        _ = f.readline()
        while line := f.readline():
            user_id, movie_id, rating, _ = line.split(",")
            if np.random.rand() < test_split:
                train.register_entry(user_id, movie_id)
                test.add_entry(user_id, movie_id, float(rating))
            else:
                train.add_entry(user_id, movie_id, float(rating))
                test.register_entry(user_id, movie_id)

    train.convert()
    test.convert()

    return (train, test)
