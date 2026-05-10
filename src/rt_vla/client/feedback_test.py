from __future__ import annotations

import numpy as np
import pytest

import rt_vla.client.feedback as feedback_module
from rt_vla.client.feedback import KeyboardFeedbackConfig
from rt_vla.client.feedback import KeyboardFeedbackProvider
from rt_vla.client.feedback import LeaderArmActionOverride
from rt_vla.client.feedback import NoopFeedbackProvider


class _FakeKeyEvent:
    def __init__(self, char: str | None = None) -> None:
        self.char = char


class _FakeListener:
    def __init__(self, *, on_press) -> None:
        self._on_press = on_press
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _FakeKeyboardBackend:
    class Key:
        esc = object()

    def __init__(self) -> None:
        self.listener: _FakeListener | None = None

    def Listener(self, on_press):  # noqa: N802
        self.listener = _FakeListener(on_press=on_press)
        return self.listener


class _FakeLeaderArm:
    def __init__(self, joint_pos: list[float], eef_pos: float) -> None:
        self.joint_pos = np.asarray(joint_pos, dtype=np.float32)
        self.eef_pos = float(eef_pos)
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_joint_pos(self) -> np.ndarray:
        return self.joint_pos

    def get_eef_pos(self) -> list[float]:
        return [self.eef_pos]


def test_noop_feedback_provider() -> None:
    provider = NoopFeedbackProvider()

    provider.before_episode()
    action, intervened = provider.override_action(
        np.asarray([1.0, 2.0], dtype=np.float32), np.zeros((2,), dtype=np.float32)
    )
    reward, terminate, reason = provider.consume_step_feedback()

    np.testing.assert_allclose(action, [1.0, 2.0])
    assert not intervened
    assert reward == 0.0
    assert not terminate
    assert reason is None


def test_keyboard_feedback_provider_requires_pynput() -> None:
    original_keyboard = feedback_module.keyboard
    feedback_module.keyboard = None
    try:
        with pytest.raises(RuntimeError, match="pynput is required for KeyboardFeedbackProvider"):
            KeyboardFeedbackProvider(keyboard_backend=None)
    finally:
        feedback_module.keyboard = original_keyboard


def test_keyboard_feedback_provider_holds_current_state_without_override_callback() -> None:
    keyboard_backend = _FakeKeyboardBackend()
    provider = KeyboardFeedbackProvider(
        keyboard_backend=keyboard_backend,
        config=KeyboardFeedbackConfig(key_debounce_sec=0.0),
    )

    provider.before_episode()
    provider.handle_key_press(_FakeKeyEvent("s"))
    policy_action = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    action, intervened = provider.override_action(policy_action, np.asarray([9.0, 8.0, 7.0], dtype=np.float32))

    np.testing.assert_allclose(action, [9.0, 8.0, 7.0])
    assert intervened
    assert keyboard_backend.listener is not None
    assert keyboard_backend.listener.started

    provider.close()
    assert keyboard_backend.listener.stopped


def test_keyboard_feedback_provider_uses_override_callback() -> None:
    keyboard_backend = _FakeKeyboardBackend()

    def _override(policy_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        del policy_action
        return current_state + 5.0

    provider = KeyboardFeedbackProvider(
        keyboard_backend=keyboard_backend,
        config=KeyboardFeedbackConfig(key_debounce_sec=0.0),
        action_override_callback=_override,
    )

    provider.before_episode()
    provider.handle_key_press(_FakeKeyEvent("s"))
    action, intervened = provider.override_action(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0, 4.0], dtype=np.float32),
    )

    np.testing.assert_allclose(action, [8.0, 9.0])
    assert intervened


def test_keyboard_feedback_provider_does_not_mark_intervention_when_callback_declines() -> None:
    keyboard_backend = _FakeKeyboardBackend()

    def _override(policy_action: np.ndarray, current_state: np.ndarray) -> None:
        del policy_action, current_state
        return None

    provider = KeyboardFeedbackProvider(
        keyboard_backend=keyboard_backend,
        config=KeyboardFeedbackConfig(key_debounce_sec=0.0),
        action_override_callback=_override,
    )

    provider.before_episode()
    provider.handle_key_press(_FakeKeyEvent("s"))
    policy_action = np.asarray([1.0, 2.0], dtype=np.float32)
    action, intervened = provider.override_action(policy_action, np.asarray([3.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(action, policy_action)
    assert not intervened


def test_keyboard_feedback_provider_consumes_reward_and_terminate_flags() -> None:
    keyboard_backend = _FakeKeyboardBackend()
    provider = KeyboardFeedbackProvider(
        keyboard_backend=keyboard_backend,
        config=KeyboardFeedbackConfig(key_debounce_sec=0.0),
    )

    provider.before_episode()
    provider.handle_key_press(_FakeKeyEvent("y"))
    provider.handle_key_press(_FakeKeyEvent("n"))
    reward, terminate, reason = provider.consume_step_feedback()

    assert reward == 1.0
    assert terminate
    assert reason == "operator_terminate"

    reward, terminate, reason = provider.consume_step_feedback()
    assert reward == 0.0
    assert not terminate
    assert reason is None


def test_keyboard_feedback_provider_abort_sets_terminal_reason() -> None:
    keyboard_backend = _FakeKeyboardBackend()
    provider = KeyboardFeedbackProvider(
        keyboard_backend=keyboard_backend,
        config=KeyboardFeedbackConfig(key_debounce_sec=0.0),
    )

    provider.before_episode()
    provider.handle_key_press(keyboard_backend.Key.esc)
    reward, terminate, reason = provider.consume_step_feedback()

    assert reward == 0.0
    assert terminate
    assert reason == "abort"


def test_leader_arm_action_override_matches_keyboard_leader_delta_mapping() -> None:
    arms = {
        50050: _FakeLeaderArm([10, 11, 12, 13, 14, 15], 0.03),
        50052: _FakeLeaderArm([20, 21, 22, 23, 24, 25], 0.04),
    }
    override = LeaderArmActionOverride(
        left_leader_port=50050,
        right_leader_port=50052,
        arm_factory=lambda port: arms[port],
    )

    current_state = np.asarray(
        [1, 2, 3, 4, 5, 6, 0.1, 7, 8, 9, 10, 11, 12, 0.2],
        dtype=np.float32,
    )
    policy_action = np.zeros((14,), dtype=np.float32)
    first_action = override(policy_action, current_state)
    np.testing.assert_allclose(first_action, [1, 2, 3, 4, 5, 6, 0.03, 7, 8, 9, 10, 11, 12, 0.04])

    arms[50050].joint_pos += np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
    arms[50052].joint_pos += np.asarray([0, 0, 0, 0, -0.25, 0], dtype=np.float32)
    second_action = override(policy_action, current_state)
    np.testing.assert_allclose(second_action, [1.5, 2, 3, 4, 5, 6, 0.03, 7, 8, 9, 10, 10.75, 12, 0.04])

    override.reset()
    reset_action = override(policy_action, current_state)
    np.testing.assert_allclose(reset_action, [1, 2, 3, 4, 5, 6, 0.03, 7, 8, 9, 10, 11, 12, 0.04])

    override.close()
    assert arms[50050].connected
    assert arms[50052].connected
    assert arms[50050].disconnected
    assert arms[50052].disconnected
