from utils.dataset import Ratings


def test_train_test_split(train: Ratings, test: Ratings, test_split: float):
    # Check for index inconsistency
    assert test.movie_id_to_idx == train.movie_id_to_idx, (
        "Inconsistent movie_id indexing"
    )
    assert test.user_id_to_idx == train.user_id_to_idx, "Inconsistent user_id indexing"
    assert len(test.user_movies) == len(train.user_movies), (
        "Inconsistent user_movies data"
    )
    assert len(test.movie_users) == len(train.movie_users), (
        "Inconsistent movie_users data"
    )
    assert ((len(test) / (len(train) + len(test))) - test_split) < 1e-2, (
        "Test split inconsistent"
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
