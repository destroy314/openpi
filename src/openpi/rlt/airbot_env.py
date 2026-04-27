import dataclasses
import json
import logging
import os
import pathlib
import queue
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

try:
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover
    keyboard = None

try:
    from airbot_py.arm import AIRBOTArm
    from airbot_py.arm import RobotMode
    from airbot_py.arm import SpeedProfile
except ImportError:  # pragma: no cover
    AIRBOTArm = None
    RobotMode = None
    SpeedProfile = None


LOGGER = logging.getLogger(__name__)

# _DEFAULT_LEFT_RESET = (-0.05, -0.26, 0.56, 1.48, -1.19, -1.34, 0.0)
# _DEFAULT_RIGHT_RESET = (-0.05, -0.26, 0.56, -1.48, 1.19, 1.34, 0.0)
_DEFAULT_LEFT_RESET  = (-0.3736, -0.8108,  0.6645,  1.4765, -0.8911, -1.3767,  0.0)
_DEFAULT_RIGHT_RESET = ( 0.3809, -0.8669,  0.7577, -1.5047,  0.9221,  1.6538,  0.0)


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    index: int | None
    width: int = 640
    height: int = 480
    fps: int = 30
    exposure: float | None = None
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0
    fake: bool = False


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    name: str
    follower_port: int
    leader_port: int | None = None
    reset_joint_pos: tuple[float, ...] = _DEFAULT_LEFT_RESET
    gripper_max_length: float = 0.07
    gripper_sleep_sec: float = 0.6
    speed_fast: bool = False

    def __post_init__(self) -> None:
        if len(self.reset_joint_pos) != 7:
            raise ValueError(f"{self.name}: reset_joint_pos must have 7 elements, got {len(self.reset_joint_pos)}")


@dataclasses.dataclass(frozen=True)
class KeyboardConfig:
    intervention_toggle_key: str = "s"
    reward_key: str = "y"
    terminate_episode_key: str = "n"
    abort_key: str = "esc"
    key_debounce_sec: float = 0.2


@dataclasses.dataclass(frozen=True)
class AirbotRLTEnvConfig:
    prompt: str = "do the task"
    control_hz: int = 25
    max_episode_steps: int = 500
    display_images: bool = True
    fake_env: bool = False
    enable_intervention: bool = True
    terminate_on_success_reward: bool = False
    post_reset_sleep_sec: float = 0.2
    keyboard: KeyboardConfig = dataclasses.field(default_factory=KeyboardConfig)
    left_arm: ArmConfig = dataclasses.field(
        default_factory=lambda: ArmConfig(name="left", follower_port=50051, leader_port=50050)
    )
    right_arm: ArmConfig = dataclasses.field(
        default_factory=lambda: ArmConfig(
            name="right",
            follower_port=50053,
            leader_port=50052,
            reset_joint_pos=_DEFAULT_RIGHT_RESET,
        )
    )
    cameras: dict[str, CameraConfig] = dataclasses.field(
        default_factory=lambda: {
            "cam_high": CameraConfig(index=4),
            "cam_left_wrist": CameraConfig(index=2),
            "cam_right_wrist": CameraConfig(index=0),
        }
    )

    @classmethod
    def from_env(cls) -> "AirbotRLTEnvConfig":
        config_path = os.environ.get("OPENPI_AIRBOT_RLT_CONFIG")
        config = cls.from_path(config_path) if config_path else cls()
        prompt_override = os.environ.get("OPENPI_AIRBOT_PROMPT")
        fake_override = os.environ.get("OPENPI_AIRBOT_FAKE_ENV")
        if prompt_override:
            config = dataclasses.replace(config, prompt=prompt_override)
        if fake_override is not None:
            fake_value = fake_override.lower() in {"1", "true", "yes", "on"}
            config = dataclasses.replace(config, fake_env=fake_value)
        return config

    @classmethod
    def from_path(cls, path: str | os.PathLike[str] | None) -> "AirbotRLTEnvConfig":
        if path is None:
            return cls()
        path = pathlib.Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Airbot RLT config file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text())
        elif suffix in {".toml", ".tml"}:
            if tomllib is None:  # pragma: no cover
                raise RuntimeError("TOML config loading requires Python 3.11+")
            payload = tomllib.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported Airbot config format: {path.suffix}. Use .json or .toml.")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AirbotRLTEnvConfig":
        defaults = dataclasses.asdict(cls())
        merged = _deep_merge(defaults, payload)
        keyboard_config = KeyboardConfig(**merged["keyboard"])
        left_arm = ArmConfig(**merged["left_arm"])
        right_arm = ArmConfig(**merged["right_arm"])
        camera_configs = {name: CameraConfig(**cfg) for name, cfg in merged["cameras"].items()}
        return cls(
            prompt=merged["prompt"],
            control_hz=int(merged["control_hz"]),
            max_episode_steps=int(merged["max_episode_steps"]),
            display_images=bool(merged["display_images"]),
            fake_env=bool(merged["fake_env"]),
            enable_intervention=bool(merged["enable_intervention"]),
            terminate_on_success_reward=bool(merged["terminate_on_success_reward"]),
            post_reset_sleep_sec=float(merged["post_reset_sleep_sec"]),
            keyboard=keyboard_config,
            left_arm=left_arm,
            right_arm=right_arm,
            cameras=camera_configs,
        )


@dataclasses.dataclass(frozen=True)
class ArmObservation:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torque: np.ndarray
    ee_pose: np.ndarray
    gripper_pos: float
    gripper_vel: float = 0.0

    def position_vector(self) -> np.ndarray:
        return np.concatenate([self.joint_pos.astype(np.float32), np.asarray([self.gripper_pos], dtype=np.float32)])

    def velocity_vector(self) -> np.ndarray:
        return np.concatenate([self.joint_vel.astype(np.float32), np.asarray([self.gripper_vel], dtype=np.float32)])


@dataclasses.dataclass(frozen=True)
class OperatorFeedback:
    reward: float = 0.0
    terminate_episode: bool = False
    terminal_reason: str | None = None


class AirbotRLTEnv:
    """Dual-arm Airbot environment for Stage 2 online RLT training.

    Controls:
    - `s`: toggle operator intervention using the leader arms.
    - `y`: emit a sparse `+1` reward on the current low-level step.
    - `n`: end the current episode with reward `0`.
    - `Esc`: abort the episode immediately with reward `0`.
    """

    def __init__(
        self,
        config: AirbotRLTEnvConfig,
        *,
        left_arm: "_FollowerArm | None" = None,
        right_arm: "_FollowerArm | None" = None,
        cameras: "Mapping[str, _AsyncVideoCapture] | None" = None,
        operator: "KeyboardLeaderOperator | None" = None,
    ) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._left_arm = left_arm or _FollowerArm(config.left_arm, fake_env=config.fake_env)
        self._right_arm = right_arm or _FollowerArm(config.right_arm, fake_env=config.fake_env)
        self._cameras = dict(cameras or self._create_cameras())
        self._operator = operator or KeyboardLeaderOperator(
            left_arm=config.left_arm,
            right_arm=config.right_arm,
            keyboard_config=config.keyboard,
            enable_intervention=config.enable_intervention and not config.fake_env,
            enable_keyboard=not config.fake_env,
            terminate_on_success_reward=config.terminate_on_success_reward,
        )
        self._image_display = _ImageDisplayThread() if config.display_images else None
        self._episode_steps = 0
        self._last_state = np.zeros((14,), dtype=np.float32)
        self._closed = False

    def reset(self) -> dict[str, Any]:
        self._require_open()
        self._operator.prepare_for_episode()
        left_future = self._executor.submit(self._left_arm.reset)
        right_future = self._executor.submit(self._right_arm.reset)
        left_future.result()
        right_future.result()
        if self._config.post_reset_sleep_sec > 0:
            time.sleep(self._config.post_reset_sleep_sec)
        self._episode_steps = 0
        observation = self._collect_observation()
        return observation

    def step(self, action_chunk: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._require_open()
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"action_chunk must have shape (executed_steps, 14), got {chunk.shape}")

        step_rewards: list[float] = []
        executed_actions: list[np.ndarray] = []
        intervened_mask: list[bool] = []
        step_observations: list[dict[str, Any]] = []
        observation = None
        done = False
        last_reason: str | None = None
        intervened_any = False
        last_intervene_action: np.ndarray | None = None

        for action in chunk:
            step_started = time.perf_counter()
            action_to_apply, intervened = self._operator.override_action(action, self._last_state.copy())
            left_action, right_action = np.split(np.asarray(action_to_apply, dtype=np.float32), 2)
            left_future = self._executor.submit(self._left_arm.apply_action, left_action)
            right_future = self._executor.submit(self._right_arm.apply_action, right_action)
            executed_left = left_future.result()
            executed_right = right_future.result()
            executed_action = np.concatenate([executed_left, executed_right]).astype(np.float32)
            executed_actions.append(executed_action)
            intervened_mask.append(bool(intervened))

            intervened_any = intervened_any or intervened
            if intervened:
                last_intervene_action = executed_action

            elapsed = time.perf_counter() - step_started
            time.sleep(max(0.0, (1.0 / float(self._config.control_hz)) - elapsed))
            observation = self._collect_observation()
            step_observations.append(observation)
            self._episode_steps += 1

            feedback = self._operator.consume_feedback()
            step_rewards.append(float(feedback.reward))
            if feedback.terminate_episode:
                done = True
                last_reason = feedback.terminal_reason or "operator_terminate"
                break
            if self._episode_steps >= self._config.max_episode_steps:
                done = True
                last_reason = "max_episode_steps"
                break

        if observation is None:
            observation = self._collect_observation()
        info: dict[str, Any] = {
            "step_rewards": step_rewards,
            "executed_actions": np.asarray(executed_actions, dtype=np.float32),
            "intervened_mask": np.asarray(intervened_mask, dtype=bool),
            "step_observations": step_observations,
            "intervened": float(intervened_any),
            "episode_steps": float(self._episode_steps),
            "executed_steps": float(len(step_rewards)),
            "success": float(any(reward > 0.0 for reward in step_rewards)),
        }
        if last_reason is not None:
            info["terminal_reason"] = last_reason
        if last_intervene_action is not None:
            info["intervene_action"] = last_intervene_action
        return observation, float(sum(step_rewards)), done, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        _safe_close(self._operator)
        _safe_close(self._left_arm)
        _safe_close(self._right_arm)
        for camera in self._cameras.values():
            _safe_close(camera)
        if self._image_display is not None:
            self._image_display.close()

    def _collect_observation(self) -> dict[str, Any]:
        left_future = self._executor.submit(self._left_arm.observe)
        right_future = self._executor.submit(self._right_arm.observe)
        left_obs = left_future.result()
        right_obs = right_future.result()
        state = np.concatenate([left_obs.position_vector(), right_obs.position_vector()]).astype(np.float32)
        proprio = np.concatenate(
            [
                left_obs.position_vector(),
                right_obs.position_vector(),
                left_obs.velocity_vector(),
                right_obs.velocity_vector(),
            ]
        ).astype(np.float32)
        self._last_state = state

        images: dict[str, np.ndarray] = {}
        for name, camera in self._cameras.items():
            image = _read_camera(camera)
            images[name] = _apply_crop(image, self._config.cameras[name])
        if self._image_display is not None:
            self._image_display.show(images)

        return {
            "state": state,
            "proprio": proprio,
            "images": images,
            "prompt": self._config.prompt,
        }

    def _create_cameras(self) -> "dict[str, _AsyncVideoCapture]":
        return {name: _AsyncVideoCapture(_OpenCVCamera(cfg)) for name, cfg in self._config.cameras.items()}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("AirbotRLTEnv is closed")


class _FollowerArm:
    def __init__(self, config: ArmConfig, *, fake_env: bool = False) -> None:
        self._config = config
        self._fake = fake_env
        self._prev_gripper_pos: float | None = None
        self._prev_gripper_observe_at: float | None = None
        if fake_env:
            self._robot = _FakeAIRBOTArm(config)
        else:
            _require_airbot_sdk()
            self._robot = AIRBOTArm(port=config.follower_port)
            self._robot.connect()
            self._robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            profile = SpeedProfile.FAST if config.speed_fast else SpeedProfile.DEFAULT
            self._robot.set_speed_profile(profile)

    def reset(self) -> None:
        if not self._fake:
            self._robot.switch_mode(RobotMode.PLANNING_POS)
            self._robot.move_to_joint_pos(joint_pos=list(self._config.reset_joint_pos[:6]), blocking=True)
            self._robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        else:
            self._robot.move_to_joint_pos(joint_pos=list(self._config.reset_joint_pos[:6]), blocking=True)
        self._robot.servo_eef_pos(float(self._config.reset_joint_pos[6]))
        self._prev_gripper_pos = None
        self._prev_gripper_observe_at = None

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(7)
        self._robot.servo_eef_pos(float(action[6]))
        self._robot.servo_joint_pos(action[:6].tolist())
        return action

    def observe(self) -> ArmObservation:
        joint_pos = _require_array(self._robot.get_joint_pos(), 6, f"{self._config.name}.joint_pos")
        joint_vel = _require_array(self._robot.get_joint_vel(), 6, f"{self._config.name}.joint_vel")
        joint_torque = _require_array(self._robot.get_joint_eff(), 6, f"{self._config.name}.joint_torque")
        ee_pose_raw = self._robot.get_end_pose()
        if ee_pose_raw is None:
            raise RuntimeError(f"{self._config.name}: AIRBOTArm.get_end_pose() returned None")
        ee_pose = _require_array(list(ee_pose_raw[0]) + list(ee_pose_raw[1]), 7, f"{self._config.name}.ee_pose")
        gripper = _require_array(self._robot.get_eef_pos(), 1, f"{self._config.name}.gripper")[0]
        now = time.monotonic()
        gripper_vel = 0.0
        if self._prev_gripper_pos is not None and self._prev_gripper_observe_at is not None:
            dt = max(now - self._prev_gripper_observe_at, 1e-6)
            gripper_vel = float((gripper - self._prev_gripper_pos) / dt)
        self._prev_gripper_pos = float(gripper)
        self._prev_gripper_observe_at = now
        return ArmObservation(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_torque=joint_torque,
            ee_pose=ee_pose,
            gripper_pos=float(gripper),
            gripper_vel=gripper_vel,
        )

    def close(self) -> None:
        disconnect = getattr(self._robot, "disconnect", None)
        if callable(disconnect):
            disconnect()


class KeyboardLeaderOperator:
    def __init__(
        self,
        *,
        left_arm: ArmConfig,
        right_arm: ArmConfig,
        keyboard_config: KeyboardConfig,
        enable_intervention: bool,
        enable_keyboard: bool,
        terminate_on_success_reward: bool,
    ) -> None:
        self._left_arm = left_arm
        self._right_arm = right_arm
        self._keyboard_config = keyboard_config
        self._enable_intervention = enable_intervention
        self._enable_keyboard = enable_keyboard
        self._terminate_on_success_reward = terminate_on_success_reward
        self._lock = threading.Lock()
        self._intervening = False
        self._toggle_requested = False
        self._pending_reward = 0.0
        self._pending_terminate = False
        self._pending_terminal_reason: str | None = None
        self._last_key_at: dict[str, float] = {}
        self._left_leader = None
        self._right_leader = None
        self._leader_start = np.zeros((12,), dtype=np.float32)
        self._follower_start = np.zeros((12,), dtype=np.float32)
        self._listener = None

        if self._enable_intervention:
            _require_airbot_sdk()
            if left_arm.leader_port is None or right_arm.leader_port is None:
                raise ValueError("Leader ports must be configured when enable_intervention=True")
            self._left_leader = AIRBOTArm(port=left_arm.leader_port)
            self._right_leader = AIRBOTArm(port=right_arm.leader_port)
            self._left_leader.connect()
            self._right_leader.connect()
        if self._enable_keyboard:
            _require_keyboard()
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()

    def prepare_for_episode(self) -> None:
        with self._lock:
            self._intervening = False
            self._toggle_requested = False
            self._pending_reward = 0.0
            self._pending_terminate = False
            self._pending_terminal_reason = None

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._enable_intervention:
            return np.asarray(policy_action, dtype=np.float32), False

        with self._lock:
            if self._toggle_requested:
                self._intervening = not self._intervening
                self._toggle_requested = False
                if self._intervening:
                    self._leader_start = self._read_leader_joints()
                    follower_joint_start = np.concatenate(
                        [np.asarray(current_state[:6], dtype=np.float32), np.asarray(current_state[7:13], dtype=np.float32)]
                    )
                    self._follower_start = follower_joint_start.copy()
                    LOGGER.info("Operator intervention enabled")
                else:
                    LOGGER.info("Operator intervention disabled")
            if not self._intervening:
                return np.asarray(policy_action, dtype=np.float32), False

        current_leader = self._read_leader_joints()
        left_delta = current_leader[:6] - self._leader_start[:6]
        right_delta = current_leader[6:] - self._leader_start[6:]
        left_target = self._follower_start[:6] + left_delta
        right_target = self._follower_start[6:] + right_delta
        left_gripper = self._read_leader_gripper(self._left_leader)
        right_gripper = self._read_leader_gripper(self._right_leader)
        left_gripper = float(left_gripper)
        right_gripper = float(right_gripper)
        expert = np.concatenate(
            [left_target, np.asarray([left_gripper], dtype=np.float32), right_target, np.asarray([right_gripper], dtype=np.float32)]
        ).astype(np.float32)
        return expert, True

    def consume_feedback(self) -> OperatorFeedback:
        with self._lock:
            feedback = OperatorFeedback(
                reward=float(self._pending_reward),
                terminate_episode=bool(self._pending_terminate),
                terminal_reason=self._pending_terminal_reason,
            )
            self._pending_reward = 0.0
            self._pending_terminate = False
            self._pending_terminal_reason = None
            return feedback

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        for leader in (self._left_leader, self._right_leader):
            disconnect = getattr(leader, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def _on_press(self, key: Any) -> None:
        key_name = None
        if key == getattr(keyboard.Key, "esc", object()):
            key_name = "esc"
        else:
            key_name = getattr(key, "char", None)
        if key_name is None:
            return
        if not self._allow_key(key_name):
            return
        LOGGER.info("Keyboard key pressed: %s", key_name)
        with self._lock:
            if key_name == self._keyboard_config.intervention_toggle_key:
                self._toggle_requested = True
            elif key_name == self._keyboard_config.reward_key:
                self._pending_reward = 1.0
                if self._terminate_on_success_reward:
                    self._pending_terminate = True
                    self._pending_terminal_reason = "success"
            elif key_name == self._keyboard_config.terminate_episode_key:
                self._pending_terminate = True
                self._pending_terminal_reason = "operator_terminate"
            elif key_name == self._keyboard_config.abort_key:
                self._pending_terminate = True
                self._pending_terminal_reason = "abort"

    def _allow_key(self, key_name: str) -> bool:
        now = time.time()
        previous = self._last_key_at.get(key_name, 0.0)
        if now - previous < self._keyboard_config.key_debounce_sec:
            return False
        self._last_key_at[key_name] = now
        return True

    def _read_leader_joints(self) -> np.ndarray:
        left = _require_array(self._left_leader.get_joint_pos(), 6, "left_leader.joint_pos")
        right = _require_array(self._right_leader.get_joint_pos(), 6, "right_leader.joint_pos")
        return np.concatenate([left, right]).astype(np.float32)

    @staticmethod
    def _read_leader_gripper(leader: Any) -> float:
        return float(_require_array(leader.get_eef_pos(), 1, "leader.gripper")[0])


class _AsyncVideoCapture:
    def __init__(self, camera: "_OpenCVCamera") -> None:
        self._camera = camera
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._closed = False
        self._last_frame: np.ndarray | None = None
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def read(self) -> np.ndarray:
        if self._last_frame is not None and self._queue.empty():
            return self._last_frame.copy()
        frame = self._queue.get(timeout=0.2)
        self._last_frame = frame
        return frame.copy()

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=1.0)
        self._camera.close()

    def _reader(self) -> None:
        while not self._closed:
            frame = self._camera.read()
            self._last_frame = frame
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)


class _OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._fake_frame = _make_fake_camera_frame(config)
        self._capture = None
        if config.fake:
            return
        if config.index is None:
            raise ValueError("Real camera config requires an integer camera index")
        capture = cv2.VideoCapture(config.index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(config.index)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open camera index {config.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        if config.exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            capture.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            time.sleep(0.02)
            return self._fake_frame.copy()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read camera index {self._config.index}")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class _ImageDisplayThread:
    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, np.ndarray] | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def show(self, images: Mapping[str, np.ndarray]) -> None:
        frame_dict = {name: np.asarray(image) for name, image in images.items()}
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put(frame_dict)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        cv2.destroyAllWindows()

    def _run(self) -> None:
        while True:
            images = self._queue.get()
            if images is None:
                return
            tiles = []
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                if name not in images:
                    continue
                image = np.asarray(images[name])
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                label = bgr.copy()
                cv2.putText(label, name, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                tiles.append(label)
            if not tiles:
                continue
            canvas = np.concatenate([cv2.resize(tile, (320, 240)) for tile in tiles], axis=1)
            cv2.imshow("airbot_rlt", canvas)
            cv2.waitKey(1)


class _FakeAIRBOTArm:
    def __init__(self, config: ArmConfig) -> None:
        self._joint_pos = np.asarray(config.reset_joint_pos[:6], dtype=np.float32)
        self._joint_vel = np.zeros((6,), dtype=np.float32)
        self._joint_eff = np.zeros((6,), dtype=np.float32)
        self._gripper = float(config.reset_joint_pos[6])

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def switch_mode(self, mode: Any) -> None:
        del mode

    def set_speed_profile(self, profile: Any) -> None:
        del profile

    def move_to_joint_pos(self, joint_pos: list[float], blocking: bool = True) -> None:
        del blocking
        self._joint_pos = np.asarray(joint_pos, dtype=np.float32)
        self._joint_vel = np.zeros((6,), dtype=np.float32)

    def servo_joint_pos(self, joint_pos: list[float]) -> None:
        new_joint_pos = np.asarray(joint_pos, dtype=np.float32)
        self._joint_vel = new_joint_pos - self._joint_pos
        self._joint_pos = new_joint_pos

    def servo_eef_pos(self, position: float) -> None:
        self._gripper = float(position)

    def get_joint_pos(self) -> list[float]:
        return self._joint_pos.tolist()

    def get_joint_vel(self) -> list[float]:
        return self._joint_vel.tolist()

    def get_joint_eff(self) -> list[float]:
        return self._joint_eff.tolist()

    def get_eef_pos(self) -> list[float]:
        return [self._gripper]

    def get_end_pose(self) -> tuple[list[float], list[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_airbot_sdk() -> None:
    if AIRBOTArm is None or RobotMode is None or SpeedProfile is None:
        raise RuntimeError(
            "airbot_py is required for the real Airbot Stage 2 environment. "
            "Install the Airbot SDK in the rl-token environment before using openpi.rlt.airbot_env:create_env."
        )


def _require_keyboard() -> None:
    if keyboard is None:
        raise RuntimeError(
            "pynput is required for keyboard intervention/reward control. "
            "Install pynput in the rl-token environment before using the Airbot Stage 2 environment."
        )


def _require_array(value: Any, expected_dim: int, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"{name} returned None")
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_dim:
        raise RuntimeError(f"{name} must have {expected_dim} elements, got shape {array.shape}")
    return array


def _read_camera(camera: "_AsyncVideoCapture") -> np.ndarray:
    frame = np.asarray(camera.read())
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Camera must return HWC uint8 images, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _apply_crop(image: np.ndarray, config: CameraConfig) -> np.ndarray:
    top = int(config.crop_top)
    bottom = image.shape[0] - int(config.crop_bottom)
    left = int(config.crop_left)
    right = image.shape[1] - int(config.crop_right)
    if top < 0 or left < 0 or bottom <= top or right <= left:
        raise ValueError(f"Invalid crop for camera index {config.index}: {(top, bottom, left, right)}")
    return image[top:bottom, left:right].copy()


def _make_fake_camera_frame(config: CameraConfig) -> np.ndarray:
    frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    label = f"FAKECAM {config.index if config.index is not None else 'N/A'}"
    cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return frame


def _safe_close(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


__all__ = [
    "AirbotRLTEnv",
    "AirbotRLTEnvConfig",
    "ArmConfig",
    "CameraConfig",
    "KeyboardConfig",
    "OperatorFeedback",
    "create_env",
]
