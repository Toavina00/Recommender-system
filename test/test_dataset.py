import os
import zipfile
from pathlib import Path
from urllib import request

import numpy as np
from pytest import fixture

from utils.dataset import Dataset, train_test_load


def check_dataloader(data: Dataset):
    assert (
        len(data.idx_to_user_id) == len(data.user_id_to_idx) == len(data.user_movies)
    ), "Inconsistent User indexing"
    assert (
        len(data.idx_to_movie_id)
        == len(data.movie_id_to_idx)
        == len(data.movie_users)
        == len(data.movie_feat)
    ), "Inconsistent Movie indexing"
    assert (
        len(data.feat_id_to_idx) == len(data.feat_movie) == len(data.idx_to_feat_id)
    ), "Inconsistent feature indexing"

    for i in range(len(data.user_movies)):
        if len(data.user_movies[i][0]) > 0:
            assert np.min(data.user_movies[i][0]) >= 0, (
                "Inconsistent User-Movie map index"
            )
            assert np.max(data.user_movies[i][0]) < len(data.movie_users), (
                "Inconsistent User-Movie map index"
            )

    for i in range(len(data.movie_users)):
        if len(data.movie_users[i][0]) > 0:
            assert np.min(data.movie_users[i][0]) >= 0, (
                "Inconsistent Movie-User map index"
            )
            assert np.max(data.movie_users[i][0]) < len(data.user_movies), (
                "Inconsistent Movie-User map index"
            )
        if len(data.movie_feat[i]) > 0:
            assert np.min(data.movie_feat[i]) >= 0, (
                "Inconsistent Movie-Feature map index"
            )
            assert np.max(data.movie_feat[i]) < len(data.feat_movie), (
                "Inconsistent Movie-Feature map index"
            )

    for i in range(len(data.feat_movie)):
        if len(data.feat_movie[i]) > 0:
            assert np.min(data.feat_movie[i]) >= 0, (
                "Inconsistent Feature-Movie map index"
            )
            assert np.max(data.feat_movie[i]) < len(data.movie_feat), (
                "Inconsistent Feature-Movie map index"
            )


def check_train_test_split(train: Dataset, test: Dataset, test_split: float):
    # Check for index inconsistency
    assert test.movie_id_to_idx == train.movie_id_to_idx, (
        "Inconsistent movie_id indexing"
    )
    assert test.user_id_to_idx == train.user_id_to_idx, "Inconsistent user_id indexing"
    assert test.feat_id_to_idx == train.feat_id_to_idx, "Inconsistent feat_id indexing"
    assert len(test.user_movies) == len(train.user_movies), (
        "Inconsistent user_movies data"
    )
    assert len(test.movie_users) == len(train.movie_users), (
        "Inconsistent movie_users data"
    )
    assert len(test.feat_movie) == len(train.feat_movie), "Inconsistent feat_move data"
    assert len(test.movie_feat) == len(train.movie_feat), "Inconsistent movie_feat data"
    assert ((len(test) / (len(train) + len(test))) - test_split) < 1e-2, (
        "Test split inconsistent"
    )

    for i in range(len(test.feat_movie)):
        assert set(test.feat_movie[i].tolist()) == set(train.feat_movie[i].tolist()), (  # pyright: ignore
            "Inconsistent feat_move data"
        )

    for i in range(len(test.movie_feat)):
        assert set(test.movie_feat[i].tolist()) == set(train.movie_feat[i].tolist()), (  # pyright: ignore
            "Inconsistent feat_move data"
        )

    # Check for train test overlap
    for r in range(len(train.movie_users)):
        assert (
            len(
                set(zip(*test.movie_users[r])).intersection(
                    set(zip(*train.movie_users[r]))
                )
            )
            == 0
        ), "Train test movie_users overlap"

    for r in range(len(train.user_movies)):
        assert (
            len(
                set(zip(*test.user_movies[r])).intersection(
                    set(zip(*train.user_movies[r]))
                )
            )
            == 0
        ), "Train test user_movies overlap"


@fixture
def data_path():
    folder = Path("data")
    data_folder = folder / "ml-latest-small"
    rating_path = data_folder / "ratings.csv"
    movies_path = data_folder / "movies.csv"

    if rating_path.exists() and movies_path.exists():
        return (rating_path.resolve(), movies_path.resolve())

    data_folder.mkdir(parents=True, exist_ok=True)

    data_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    temp_zip = folder / "ml-latest-small.zip"

    if temp_zip.exists():
        os.remove(temp_zip.resolve())

    try:
        request.urlretrieve(data_url, temp_zip.resolve())
        with zipfile.ZipFile(temp_zip.resolve()) as zip:
            zip.extractall(folder.resolve())

    except Exception as e:
        print(f"Error: {e}")

    finally:
        os.remove(temp_zip.resolve())

    if (not rating_path.exists()) or (not movies_path.exists()):
        raise Exception("Could not find movielens dataset for test")

    return (rating_path.resolve(), movies_path.resolve())


def test_dataloader(data_path):
    test_split = 0.2

    train, test = train_test_load(data_path[0], data_path[1], test_split)

    print(len(train))

    check_dataloader(train)
    check_dataloader(test)

    check_train_test_split(train, test, test_split)
