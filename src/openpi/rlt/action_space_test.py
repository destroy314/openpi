import numpy as np
import pytest

from openpi.rlt import action_space
from openpi.shared.normalize import NormStats


def _make_norm_stats(dim: int) -> dict[str, NormStats]:
    return {
        "actions": NormStats(
            mean=np.linspace(1.0, float(dim), dim, dtype=np.float32),
            std=np.linspace(0.5, 1.5, dim, dtype=np.float32),
            q01=np.linspace(-2.0, -1.0, dim, dtype=np.float32),
            q99=np.linspace(2.0, 3.0, dim, dtype=np.float32),
        )
    }


def test_normalize_denormalize_roundtrip_with_delta_joint_actions():
    env_dim = 14
    stats = _make_norm_stats(env_dim)
    state = np.linspace(10.0, 23.0, env_dim, dtype=np.float32)
    absolute_actions = np.stack(
        [
            state + np.linspace(0.0, 1.3, env_dim, dtype=np.float32),
            state + np.linspace(1.0, 2.3, env_dim, dtype=np.float32),
        ],
        axis=0,
    )

    normalized = action_space.normalize_action(
        stats,
        use_quantiles=False,
        actions=absolute_actions,
        state=state,
        use_delta_joint_actions=True,
    )
    restored = action_space.denormalize_action(
        stats,
        use_quantiles=False,
        actions=normalized,
        state=state,
        use_delta_joint_actions=True,
    )

    np.testing.assert_allclose(restored, absolute_actions, rtol=1e-5, atol=1e-5)


def test_delta_joint_actions_require_state():
    stats = _make_norm_stats(14)
    actions = np.zeros((2, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="state is required"):
        action_space.normalize_action(stats, False, actions, use_delta_joint_actions=True)

    with pytest.raises(ValueError, match="state is required"):
        action_space.denormalize_action(stats, False, actions, use_delta_joint_actions=True)


def test_action_transform_state_validates_dimension():
    with pytest.raises(ValueError, match="state must have at least 14 dims, got 13"):
        action_space.action_transform_state(np.zeros((13,), dtype=np.float32), 14)


def test_slice_norm_stats_truncates_all_fields():
    stats = NormStats(
        mean=np.arange(20, dtype=np.float32),
        std=np.arange(20, dtype=np.float32) + 1.0,
        q01=np.arange(20, dtype=np.float32) - 1.0,
        q99=np.arange(20, dtype=np.float32) + 2.0,
    )

    sliced = action_space.slice_norm_stats(stats, 14)

    assert sliced.mean.shape == (14,)
    assert sliced.std.shape == (14,)
    assert sliced.q01 is not None and sliced.q01.shape == (14,)
    assert sliced.q99 is not None and sliced.q99.shape == (14,)
    np.testing.assert_array_equal(sliced.mean, stats.mean[:14])
    np.testing.assert_array_equal(sliced.std, stats.std[:14])
    np.testing.assert_array_equal(sliced.q01, stats.q01[:14])
    np.testing.assert_array_equal(sliced.q99, stats.q99[:14])
