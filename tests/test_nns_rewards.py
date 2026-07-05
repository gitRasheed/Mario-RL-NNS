from mario_rl_nns.nns_rewards import NNSRewardConfig, NNSRewardState, lpm, upm


def test_partial_moments_are_target_relative() -> None:
    assert lpm(0.25, target=0.5, degree=2.0) == 0.0625
    assert upm(0.75, target=0.5, degree=1.0) == 0.25
    assert lpm(0.75, target=0.5, degree=2.0) == 0.0
    assert upm(0.25, target=0.5, degree=1.0) == 0.0


def test_nns_reward_keeps_base_and_train_reward_separate() -> None:
    state = NNSRewardState(NNSRewardConfig(window=2, target_speed=1.0))
    state.reset(0.0)
    shaped = state.shape(1.0, progress=0.0, death=True)
    assert shaped["reward_base"] == 1.0
    assert shaped["reward_train"] < shaped["reward_base"]
    assert shaped["nns_extra_reward"] >= -5.0


def test_nns_warmup_disables_shaping_temporarily() -> None:
    state = NNSRewardState(
        NNSRewardConfig(window=2, target_speed=1.0, global_warmup_steps=1, n_envs=1)
    )
    state.reset(0.0)
    warmup = state.shape(1.0, progress=0.0, death=True)
    shaped = state.shape(1.0, progress=0.0, death=True)
    assert warmup["reward_train"] == warmup["reward_base"]
    assert warmup["warmup_active"] == 1.0
    assert shaped["reward_train"] < shaped["reward_base"]
    assert shaped["warmup_active"] == 0.0
