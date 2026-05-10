from __future__ import annotations

import numpy as np

from rt_vla.server.model import OpenPiRTCTritonAdapter


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward(self, observation_images, diffusion_noise, **kwargs):
        self.calls.append(
            {
                "observation_images": observation_images,
                "diffusion_noise": diffusion_noise,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeTensor(np.arange(4 * 32, dtype=np.float32).reshape(4, 32))


def test_triton_adapter_samples_pi05_without_rtc_prefill_or_inpainting() -> None:
    adapter = OpenPiRTCTritonAdapter.__new__(OpenPiRTCTritonAdapter)
    adapter._key_mapping = {}
    adapter._policy = _RecordingPolicy()
    adapter.prompt = "pick up block"
    adapter.state_dim = 14
    adapter.action_dim = 14
    adapter.valid_action_num = 2

    seen: dict[str, np.ndarray] = {}

    adapter.process_images = lambda _state, _image_type: None
    adapter._build_observation_images = lambda: "images"
    adapter._build_diffusion_noise = lambda: "noise"
    adapter._normalize_state = lambda obs_state, target_dim=32: seen.setdefault(
        "obs_state",
        np.asarray(obs_state, dtype=np.float32).copy(),
    ) * 0.0
    adapter._digitize_state = lambda state_normed: np.asarray(state_normed[:14], dtype=np.int32)
    adapter._unnormalize_actions = lambda actions, target_dim=32: np.asarray(actions, dtype=np.float32)
    adapter._to_absolute_actions = lambda actions, obs_state: np.asarray(actions, dtype=np.float32)
    adapter.process_actions_for_robot = lambda actions: np.asarray(actions, dtype=np.float32)

    result = adapter.infer_actions(
        {
            "state": np.arange(14, dtype=np.float32),
            "action": np.full((3, 14), 99.0, dtype=np.float32),
        }
    )

    np.testing.assert_allclose(seen["obs_state"], np.arange(14, dtype=np.float32))
    assert len(adapter._policy.calls) == 1
    assert set(adapter._policy.calls[0]["kwargs"]) == {"task_prompt", "state_tokens"}
    np.testing.assert_allclose(np.asarray(result), np.arange(2 * 32, dtype=np.float32).reshape(2, 32)[:, :14])
