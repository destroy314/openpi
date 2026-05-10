from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
import dataclasses
import logging
import threading
import time
from typing import Any
from typing import Protocol

import numpy as np

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover
    keyboard = None


LOGGER = logging.getLogger(__name__)

ActionOverrideCallback = Callable[[np.ndarray, np.ndarray], np.ndarray | None]
LeaderArmFactory = Callable[[int], Any]


class FeedbackProvider(Protocol):
    def before_episode(self) -> None: ...

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]: ...

    def consume_step_feedback(self) -> tuple[float, bool, str | None]: ...

    def close(self) -> None: ...


class BaseFeedbackProvider(ABC):
    @abstractmethod
    def before_episode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        raise NotImplementedError

    @abstractmethod
    def consume_step_feedback(self) -> tuple[float, bool, str | None]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NoopFeedbackProvider(BaseFeedbackProvider):
    def before_episode(self) -> None:
        return None

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        del current_state
        return np.asarray(policy_action, dtype=np.float32), False

    def consume_step_feedback(self) -> tuple[float, bool, str | None]:
        return 0.0, False, None


@dataclasses.dataclass(frozen=True)
class KeyboardFeedbackConfig:
    intervention_toggle_key: str = "s"
    reward_key: str = "y"
    terminate_episode_key: str = "n"
    abort_key: str = "esc"
    key_debounce_sec: float = 0.2


class LeaderArmActionOverride:
    def __init__(
        self,
        *,
        left_leader_port: int,
        right_leader_port: int,
        arm_factory: LeaderArmFactory | None = None,
    ) -> None:
        factory = _create_airbot_leader_arm if arm_factory is None else arm_factory
        self._left_leader = factory(int(left_leader_port))
        self._right_leader = factory(int(right_leader_port))
        for leader in (self._left_leader, self._right_leader):
            connect = getattr(leader, "connect", None)
            if callable(connect):
                connect()
        self._leader_start: np.ndarray | None = None
        self._follower_start: np.ndarray | None = None

    def __call__(self, policy_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        policy_action_array = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        current_state_array = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if policy_action_array.size != 14:
            raise ValueError(f"Leader override requires 14D policy action, got {policy_action_array.size}")
        if current_state_array.size < 14:
            raise ValueError(
                f"Leader override requires current_state with at least 14 values, got {current_state_array.size}"
            )

        if self._leader_start is None or self._follower_start is None:
            self._leader_start = self._read_leader_joints()
            self._follower_start = np.concatenate([current_state_array[:6], current_state_array[7:13]]).astype(
                np.float32
            )

        current_leader = self._read_leader_joints()
        left_delta = current_leader[:6] - self._leader_start[:6]
        right_delta = current_leader[6:] - self._leader_start[6:]
        left_target = self._follower_start[:6] + left_delta
        right_target = self._follower_start[6:] + right_delta
        left_gripper = self._read_leader_gripper(self._left_leader)
        right_gripper = self._read_leader_gripper(self._right_leader)
        return np.concatenate(
            [
                left_target,
                np.asarray([left_gripper], dtype=np.float32),
                right_target,
                np.asarray([right_gripper], dtype=np.float32),
            ]
        ).astype(np.float32)

    def reset(self) -> None:
        self._leader_start = None
        self._follower_start = None

    def close(self) -> None:
        for leader in (self._left_leader, self._right_leader):
            disconnect = getattr(leader, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def _read_leader_joints(self) -> np.ndarray:
        left = _require_array(self._left_leader.get_joint_pos(), 6, "left_leader.joint_pos")
        right = _require_array(self._right_leader.get_joint_pos(), 6, "right_leader.joint_pos")
        return np.concatenate([left, right]).astype(np.float32)

    @staticmethod
    def _read_leader_gripper(leader: Any) -> float:
        return float(_require_array(leader.get_eef_pos(), 1, "leader.gripper")[0])


class KeyboardFeedbackProvider(BaseFeedbackProvider):
    def __init__(
        self,
        *,
        config: KeyboardFeedbackConfig | None = None,
        action_override_callback: ActionOverrideCallback | None = None,
        keyboard_backend: Any = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._config = KeyboardFeedbackConfig() if config is None else config
        self._action_override_callback = action_override_callback
        self._time_fn = time_fn
        self._keyboard = keyboard if keyboard_backend is None else keyboard_backend
        self._lock = threading.Lock()
        self._intervening = False
        self._toggle_requested = False
        self._pending_reward = 0.0
        self._pending_terminate = False
        self._pending_terminal_reason: str | None = None
        self._last_key_at: dict[str, float] = {}
        self._listener = None

        _require_keyboard(self._keyboard)
        self._listener = self._keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def before_episode(self) -> None:
        with self._lock:
            self._intervening = False
            self._toggle_requested = False
            self._pending_reward = 0.0
            self._pending_terminate = False
            self._pending_terminal_reason = None
            self._last_key_at.clear()
        self._reset_action_override_callback()

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        policy_action_array = np.asarray(policy_action, dtype=np.float32)
        current_state_array = np.asarray(current_state, dtype=np.float32)

        reset_override_callback = False
        with self._lock:
            if self._toggle_requested:
                self._intervening = not self._intervening
                self._toggle_requested = False
                reset_override_callback = True
                LOGGER.info("Operator intervention %s", "enabled" if self._intervening else "disabled")
            intervening = self._intervening

        if reset_override_callback:
            self._reset_action_override_callback()

        if not intervening:
            return policy_action_array, False

        if self._action_override_callback is None:
            if current_state_array.shape == policy_action_array.shape:
                return current_state_array.copy(), True
            return policy_action_array, False

        override_action = self._action_override_callback(policy_action_array.copy(), current_state_array.copy())
        if override_action is None:
            return policy_action_array, False
        return np.asarray(override_action, dtype=np.float32), True

    def consume_step_feedback(self) -> tuple[float, bool, str | None]:
        with self._lock:
            feedback = (
                float(self._pending_reward),
                bool(self._pending_terminate),
                self._pending_terminal_reason,
            )
            self._pending_reward = 0.0
            self._pending_terminate = False
            self._pending_terminal_reason = None
        return feedback

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        close = getattr(self._action_override_callback, "close", None)
        if callable(close):
            close()

    def _on_press(self, key: Any) -> None:
        self.handle_key_press(key)

    def handle_key_press(self, key: Any) -> None:
        key_name = "esc" if key == getattr(self._keyboard.Key, "esc", object()) else getattr(key, "char", None)
        if key_name is None or not self._allow_key(key_name):
            return

        LOGGER.info("Keyboard key pressed: %s", key_name)
        with self._lock:
            if key_name == self._config.intervention_toggle_key:
                self._toggle_requested = True
            elif key_name == self._config.reward_key:
                self._pending_reward = 1.0
            elif key_name == self._config.terminate_episode_key:
                self._pending_terminate = True
                self._pending_terminal_reason = "operator_terminate"
            elif key_name == self._config.abort_key:
                self._pending_terminate = True
                self._pending_terminal_reason = "abort"

    def _allow_key(self, key_name: str) -> bool:
        now = self._time_fn()
        previous = self._last_key_at.get(key_name, 0.0)
        if now - previous < self._config.key_debounce_sec:
            return False
        self._last_key_at[key_name] = now
        return True

    def _reset_action_override_callback(self) -> None:
        reset = getattr(self._action_override_callback, "reset", None)
        if callable(reset):
            reset()


def _require_keyboard(keyboard_backend: Any) -> None:
    if keyboard_backend is None:
        raise RuntimeError(
            "pynput is required for KeyboardFeedbackProvider. "
            "Install pynput or use NoopFeedbackProvider when keyboard feedback is unavailable."
        )


def _create_airbot_leader_arm(port: int) -> Any:
    try:
        from airbot_py.arm import AIRBOTArm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "airbot_py is required for leader-arm override. "
            "Disable feedback.enable_leader_override or install the Airbot SDK."
        ) from exc
    arm = AIRBOTArm(port=int(port))
    return arm


def _require_array(value: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < size:
        raise ValueError(f"{name} must contain at least {size} values, got shape {arr.shape}")
    return arr[:size]
