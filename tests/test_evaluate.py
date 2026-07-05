from mario_rl_nns.evaluate import _parse_int_list, _summary


def test_eval_summary_counts_unique_trajectories() -> None:
    rows = [
        {
            "clear": False,
            "death": True,
            "timeout": False,
            "eval_cap": False,
            "max_progress": 1,
            "episode_length": 2,
            "episode_return_train": 3,
            "trajectory_hash": "a",
        },
        {
            "clear": False,
            "death": True,
            "timeout": False,
            "eval_cap": False,
            "max_progress": 5,
            "episode_length": 4,
            "episode_return_train": 6,
            "trajectory_hash": "b",
        },
        {
            "clear": False,
            "death": True,
            "timeout": False,
            "eval_cap": False,
            "max_progress": 5,
            "episode_length": 4,
            "episode_return_train": 6,
            "trajectory_hash": "b",
        },
    ]
    summary = _summary(rows)
    assert summary["unique_trajectory_count"] == 2
    assert summary["max_progress_p50"] == 5


def test_parse_int_list_allows_empty_default() -> None:
    assert _parse_int_list(None) is None
    assert _parse_int_list("0, 5,10") == [0, 5, 10]
