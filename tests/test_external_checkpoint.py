from mario_rl_nns.external_checkpoint import _steps


def test_steps_parses_comma_list() -> None:
    assert _steps("250000, 500000,1000000") == [250000, 500000, 1000000]
