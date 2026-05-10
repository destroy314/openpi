import numpy as np
import pytest

from rt_vla.client.rlt_observation import ObservationAdapterConfig
from rt_vla.client.rlt_observation import RLTObservationAdapter
from rt_vla.client.rlt_observation import VelocityFilterConfig


def _observer_sample(**overrides: object) -> dict[str, object]:
    sample: dict[str, object] = {
        "state": np.arange(14, dtype=np.float32),
        "timestamp": 10.0,
        "images": {
            "high": b"high",
            "left_hand": b"left",
            "right_hand": b"right",
        },
    }
    sample.update(overrides)
    return sample


def test_adapter_remaps_images_and_builds_proprio_from_sdk_velocity() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="sdk"))

    adapted = adapter.adapt(_observer_sample(joint_velocity=np.ones((14,), dtype=np.float32)))

    assert adapted.images == {
        "cam_high": b"high",
        "cam_left_wrist": b"left",
        "cam_right_wrist": b"right",
    }
    np.testing.assert_allclose(adapted.state, np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(adapted.velocity, np.ones((14,), dtype=np.float32))
    np.testing.assert_allclose(adapted.proprio[:14], adapted.state)
    np.testing.assert_allclose(adapted.proprio[14:], adapted.velocity)


def test_adapter_rejects_non_14d_state() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="sdk"))

    with pytest.raises(ValueError, match="must have length 14"):
        adapter.adapt(
            _observer_sample(
                state=np.ones((13,), dtype=np.float32),
                velocity=np.zeros((14,), dtype=np.float32),
            )
        )


def test_sdk_velocity_source_requires_explicit_velocity_field() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="sdk"))

    with pytest.raises(KeyError, match="velocity_source='sdk' requires an explicit velocity field"):
        adapter.adapt(_observer_sample())


def test_sdk_velocity_timestamp_skew_raises() -> None:
    adapter = RLTObservationAdapter(
        ObservationAdapterConfig(
            velocity_source="sdk",
            filter=VelocityFilterConfig(max_timestamp_skew_sec=0.01),
        )
    )

    with pytest.raises(ValueError, match="timestamp skew"):
        adapter.adapt(
            _observer_sample(
                velocity={"value": np.zeros((14,), dtype=np.float32), "timestamp": 10.05},
            )
        )


def test_finite_difference_velocity_uses_timestamp_and_zero_initial_velocity() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="finite_difference"))

    first = adapter.adapt(_observer_sample())
    second = adapter.adapt(_observer_sample(state=np.arange(14, dtype=np.float32) + 0.2, timestamp=10.1))

    np.testing.assert_allclose(first.velocity, np.zeros((14,), dtype=np.float32))
    np.testing.assert_allclose(second.velocity, np.full((14,), 2.0, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_finite_difference_requires_increasing_timestamp() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="finite_difference"))
    adapter.adapt(_observer_sample())

    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.adapt(_observer_sample(state=np.arange(14, dtype=np.float32) + 1.0, timestamp=10.0))


def test_finite_difference_applies_clip_and_low_pass() -> None:
    adapter = RLTObservationAdapter(
        ObservationAdapterConfig(
            velocity_source="finite_difference",
            filter=VelocityFilterConfig(clip_abs=2.0, low_pass_alpha=0.5),
        )
    )

    adapter.adapt(_observer_sample())
    second = adapter.adapt(_observer_sample(state=np.arange(14, dtype=np.float32) + 10.0, timestamp=11.0))
    third = adapter.adapt(_observer_sample(state=np.arange(14, dtype=np.float32) + 14.0, timestamp=12.0))

    np.testing.assert_allclose(second.velocity, np.full((14,), 1.0, dtype=np.float32))
    np.testing.assert_allclose(third.velocity, np.full((14,), 1.5, dtype=np.float32))


def test_sdk_velocity_accepts_mapping_payload_and_field_priority() -> None:
    adapter = RLTObservationAdapter(ObservationAdapterConfig(velocity_source="sdk"))

    adapted = adapter.adapt(
        _observer_sample(
            velocity={"value": np.full((14,), 3.0, dtype=np.float32), "timestamp": 10.0},
            joint_velocity=np.full((14,), 7.0, dtype=np.float32),
        )
    )

    np.testing.assert_allclose(adapted.velocity, np.full((14,), 3.0, dtype=np.float32))
