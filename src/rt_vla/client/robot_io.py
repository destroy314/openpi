from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from openpi.rlt import airbot_env as _airbot_env

try:
    from airbot_py.arm import AIRBOTPlay
    from airbot_py.arm import RobotMode
    from airbot_py.arm import SpeedProfile
except Exception:
    AIRBOTPlay = None
    RobotMode = None
    SpeedProfile = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


logger = logging.getLogger(__name__)


class BaseActuator(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    @abstractmethod
    def apply(self, action: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class BaseObserver(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, pending_actions_provider):
        raise NotImplementedError

    @abstractmethod
    def __post_init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state_observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_image_observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def get_raw_image_observation(self) -> dict | None:
        return None

    def drop_frame(self) -> None:
        return

    def start(self) -> None:
        return


@dataclass
class NoopActuator(BaseActuator):
    @classmethod
    def from_config(cls, cfg):
        return cls()

    def __post_init__(self) -> None:
        return

    def apply(self, action: np.ndarray) -> None:
        return

    def close(self) -> None:
        return


class _AirbotRobot:
    def __init__(self, airbot_host: str, left_port: int, right_port: int):
        if AIRBOTPlay is None or RobotMode is None or SpeedProfile is None:
            raise RuntimeError("airbot_py is not available")
        self._left_arm = AIRBOTPlay(url=airbot_host, port=int(left_port))
        self._right_arm = AIRBOTPlay(url=airbot_host, port=int(right_port))
        self._connected = False
        self._previous_left_gripper: tuple[float, float] | None = None
        self._previous_right_gripper: tuple[float, float] | None = None

    def connect(self) -> None:
        if self._connected:
            return
        self._left_arm.connect()
        self._right_arm.connect()
        self._left_arm.set_speed_profile(SpeedProfile.FAST)
        self._right_arm.set_speed_profile(SpeedProfile.FAST)
        self._left_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self._right_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._left_arm.disconnect()
        self._right_arm.disconnect()
        self._connected = False

    def get_joint_state(self) -> tuple[list[float], list[float], float]:
        timestamp = time.time()
        left_pos = self._left_arm.get_joint_pos()
        left_eef_pos = self._left_arm.get_eef_pos()
        left_vel = _require_sdk_joint_velocity(self._left_arm, "left")
        right_pos = self._right_arm.get_joint_pos()
        right_eef_pos = self._right_arm.get_eef_pos()
        right_vel = _require_sdk_joint_velocity(self._right_arm, "right")
        left_gripper = float(np.asarray(left_eef_pos, dtype=np.float32).reshape(-1)[0])
        right_gripper = float(np.asarray(right_eef_pos, dtype=np.float32).reshape(-1)[0])
        state = (
            list(left_pos[:6])
            + [float(np.asarray(left_eef_pos, dtype=np.float32).reshape(-1)[0])]
            + list(right_pos[:6])
            + [float(np.asarray(right_eef_pos, dtype=np.float32).reshape(-1)[0])]
        )
        velocity = (
            list(left_vel[:6])
            + [self._estimate_gripper_velocity("left", left_gripper, timestamp)]
            + list(right_vel[:6])
            + [self._estimate_gripper_velocity("right", right_gripper, timestamp)]
        )
        return state, velocity, timestamp

    def send_action(self, action: np.ndarray, left_gripper_bias: float = 0.0, right_gripper_bias: float = 0.0) -> None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < 14:
            return
        self._left_arm.servo_joint_pos([float(v) for v in action_arr[:6]])
        self._left_arm.servo_eef_pos([float(action_arr[6]) + float(left_gripper_bias)])
        self._right_arm.servo_joint_pos([float(v) for v in action_arr[7:13]])
        self._right_arm.servo_eef_pos([float(action_arr[13]) + float(right_gripper_bias)])

    def _estimate_gripper_velocity(self, side: str, gripper: float, timestamp: float) -> float:
        # AIRBOT exposes SDK joint velocity for the six arm joints, but not for the
        # gripper/eef channel; this finite difference is the intended gripper source.
        if side == "left":
            previous = self._previous_left_gripper
            self._previous_left_gripper = (float(gripper), float(timestamp))
        else:
            previous = self._previous_right_gripper
            self._previous_right_gripper = (float(gripper), float(timestamp))
        if previous is None:
            return 0.0
        previous_gripper, previous_timestamp = previous
        dt = max(float(timestamp) - float(previous_timestamp), 1e-6)
        return float((float(gripper) - previous_gripper) / dt)


def create_airbot_robot(airbot_host: str, left_port: int, right_port: int) -> _AirbotRobot:
    return _AirbotRobot(airbot_host=airbot_host, left_port=left_port, right_port=right_port)


@dataclass
class AirbotActuator(BaseActuator):
    airbot_host: str
    left_port: int
    right_port: int
    left_gripper_bias: float = 0.0
    right_gripper_bias: float = 0.0
    robot: _AirbotRobot | None = None

    @classmethod
    def from_config(cls, cfg, robot: _AirbotRobot | None = None):
        ex = cfg.executor
        return cls(
            airbot_host=ex.airbot_host,
            left_port=ex.airbot_left_port,
            right_port=ex.airbot_right_port,
            left_gripper_bias=ex.left_gripper_bias,
            right_gripper_bias=ex.right_gripper_bias,
            robot=robot,
        )

    def __post_init__(self) -> None:
        self._robot = self.robot or create_airbot_robot(
            self.airbot_host,
            self.left_port,
            self.right_port,
        )
        self._robot.connect()

    def apply(self, action: np.ndarray) -> None:
        self._robot.send_action(
            action,
            left_gripper_bias=self.left_gripper_bias,
            right_gripper_bias=self.right_gripper_bias,
        )

    def close(self) -> None:
        self._robot.disconnect()


def _encode_jpg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image as jpg")
    return buf.tobytes()


@dataclass
class MockStateObserver(BaseObserver):
    state_dim: int
    image_size: tuple[int, int]
    pending_actions_provider: callable = None

    @classmethod
    def from_config(cls, cfg, pending_actions_provider):
        obs = cfg.observer
        return cls(
            state_dim=obs.state_dim,
            image_size=obs.image_size,
            pending_actions_provider=pending_actions_provider,
        )

    def __post_init__(self) -> None:
        self._state_step = 0
        self._image_step = 0
        self._last_state: list[float] | None = None
        self._last_timestamp: float = 0.0
        self._last_velocity: list[float] | None = None

    def _generate_state(self) -> tuple[list[float], float]:
        self._state_step += 1
        now = time.time()
        phase = self._state_step * 0.03
        state = (0.5 * np.sin(np.arange(self.state_dim, dtype=np.float32) * 0.2 + phase)).tolist()
        self._last_state = state
        self._last_timestamp = now
        return state, now

    def _estimate_velocity(self, state: list[float], timestamp: float) -> list[float]:
        if self._last_state is None or self._last_timestamp <= 0:
            velocity = np.zeros((self.state_dim,), dtype=np.float32)
        else:
            dt = max(float(timestamp) - float(self._last_timestamp), 1e-6)
            velocity = (np.asarray(state, dtype=np.float32) - np.asarray(self._last_state, dtype=np.float32)) / dt
        self._last_velocity = velocity.astype(np.float32).tolist()
        return self._last_velocity

    def _generate_raw_images(self) -> tuple[dict[str, np.ndarray], float]:
        self._image_step += 1
        now = time.time()
        h, w = self.image_size[1], self.image_size[0]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"frame={self._image_step}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"state={self._state_step}",
            (24, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 255),
            2,
        )
        return {
            "high": frame.copy(),
            "left_hand": frame.copy(),
            "right_hand": frame.copy(),
        }, now

    def get_state_observation(self) -> dict:
        previous_state = None if self._last_state is None else list(self._last_state)
        previous_timestamp = float(self._last_timestamp)
        state, state_ts = self._generate_state()
        if previous_state is None or previous_timestamp <= 0:
            joint_velocity = np.zeros((self.state_dim,), dtype=np.float32)
        else:
            dt = max(float(state_ts) - previous_timestamp, 1e-6)
            joint_velocity = (np.asarray(state, dtype=np.float32) - np.asarray(previous_state, dtype=np.float32)) / dt
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "state": state,
            "joint_velocity": joint_velocity.tolist(),
            "action": state,
            "state_timestamp": state_ts,
            "timestamp": state_ts,
            "pending_actions": pending,
        }

    def get_image_observation(self) -> dict:
        raw_images, image_ts = self._generate_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "images": {name: _encode_jpg(frame) for name, frame in raw_images.items()},
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def get_raw_image_observation(self) -> dict:
        raw_images, image_ts = self._generate_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "raw_images": raw_images,
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def close(self) -> None:
        return


class _RealSenseCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.ready = False
        self.running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self.ready = True

    @staticmethod
    def _enable_global_time(profile) -> None:
        device = profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)

    def start(self) -> None:
        if not self.ready:
            raise RuntimeError(f"Camera {self.serial} is not ready")
        if self._started:
            return
        profile = self.pipeline.start(self.config)
        self._enable_global_time(profile)
        self.running = True

        def _update_frame() -> None:
            while self.running:
                try:
                    frame = self.pipeline.wait_for_frames(timeout_ms=2000)
                except Exception as exc:
                    logger.warning("Camera %s frame update failed: %s", self.serial, exc)
                    continue
                with self._lock:
                    self._frame = frame

        self._thread = threading.Thread(target=_update_frame, daemon=True)
        self._thread.start()
        self._started = True

    def get_frame(self) -> dict | None:
        if not self.ready:
            return None
        while self.running:
            with self._lock:
                frame = self._frame
                if frame is not None:
                    self._frame = None
                    break
            time.sleep(0.01)
        else:
            return None

        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()
        if color_frame is None:
            raise RuntimeError(f"Camera {self.serial}: color frame is None")
        color_img = np.asanyarray(color_frame.get_data())
        color_timestamp = float(color_frame.get_timestamp() / 1000.0)
        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame is not None else None
        depth_timestamp = float(depth_frame.get_timestamp() / 1000.0) if depth_frame is not None else color_timestamp
        return {
            "color_image": color_img,
            "color_timestamp": color_timestamp,
            "depth_image": depth_img,
            "depth_timestamp": depth_timestamp,
        }

    def drop_frame(self) -> None:
        if not self.ready or not self._started:
            return
        with self._lock:
            self._frame = None

    def close(self) -> None:
        if not self.ready or not self._started:
            return
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.pipeline.stop()
        self._started = False


class _AirbotCameraRig:
    def __init__(
        self, top_camera_id: str, left_camera_id: str, right_camera_id: str, width: int, height: int, fps: int
    ):
        self.left_camera = _RealSenseCamera(left_camera_id, width=width, height=height, fps=fps)
        self.right_camera = _RealSenseCamera(right_camera_id, width=width, height=height, fps=fps)
        self.top_camera = _RealSenseCamera(top_camera_id, width=width, height=height, fps=fps)

    def start(self) -> None:
        self.left_camera.start()
        self.right_camera.start()
        self.top_camera.start()

    def stop(self) -> None:
        self.left_camera.close()
        self.right_camera.close()
        self.top_camera.close()

    def get_frame(self) -> dict[str, dict | None]:
        return {
            "left_camera": self.left_camera.get_frame(),
            "right_camera": self.right_camera.get_frame(),
            "top_camera": self.top_camera.get_frame(),
        }

    def drop_frame(self) -> None:
        self.left_camera.drop_frame()
        self.right_camera.drop_frame()
        self.top_camera.drop_frame()


class _RLTOpenCVCameraRig:
    _OUTPUT_KEYS = {
        "cam_high": "high",
        "cam_left_wrist": "left_hand",
        "cam_right_wrist": "right_hand",
    }

    def __init__(self, cameras: Mapping[str, _airbot_env.CameraConfig]) -> None:
        missing = [name for name in self._OUTPUT_KEYS if name not in cameras]
        if missing:
            raise ValueError(f"RLT camera config missing required cameras: {missing}")
        self._configs = {name: cameras[name] for name in self._OUTPUT_KEYS}
        self._captures = {
            name: _airbot_env._AsyncVideoCapture(_airbot_env._OpenCVCamera(config))
            for name, config in self._configs.items()
        }

    def start(self) -> None:
        return None

    def stop(self) -> None:
        for capture in self._captures.values():
            capture.close()

    def get_images(self) -> tuple[dict[str, np.ndarray], float]:
        images: dict[str, np.ndarray] = {}
        for rlt_name, output_name in self._OUTPUT_KEYS.items():
            frame = _airbot_env._read_camera(self._captures[rlt_name])
            images[output_name] = _airbot_env._apply_crop(frame, self._configs[rlt_name])
        return images, time.time()

    def drop_frame(self) -> None:
        return None


@dataclass
class AirbotRealObserver(BaseObserver):
    airbot_host: str
    left_port: int
    right_port: int
    top_camera_id: str
    left_camera_id: str
    right_camera_id: str
    enable_cameras: bool
    image_size: tuple[int, int]
    fps: int
    airbot_rlt_config_path: str = ""
    pending_actions_provider: callable = None
    robot: _AirbotRobot | None = None

    @classmethod
    def from_config(cls, cfg, pending_actions_provider, robot: _AirbotRobot | None = None):
        obs = cfg.observer
        return cls(
            airbot_host=obs.airbot_host,
            left_port=obs.airbot_left_port,
            right_port=obs.airbot_right_port,
            top_camera_id=obs.top_camera_id,
            left_camera_id=obs.left_camera_id,
            right_camera_id=obs.right_camera_id,
            enable_cameras=obs.enable_cameras,
            image_size=obs.image_size,
            fps=obs.fps,
            airbot_rlt_config_path=obs.airbot_rlt_config_path,
            pending_actions_provider=pending_actions_provider,
            robot=robot,
        )

    def __post_init__(self) -> None:
        self._robot = self.robot or create_airbot_robot(
            self.airbot_host,
            self.left_port,
            self.right_port,
        )
        self._robot.connect()
        self._camera_rig: _RLTOpenCVCameraRig | None = None
        self._cameras_started = False
        if self.enable_cameras:
            rlt_env_config = _load_airbot_rlt_env_config(self.airbot_rlt_config_path)
            self._camera_rig = _RLTOpenCVCameraRig(rlt_env_config.cameras)

    def _read_robot_state(self) -> tuple[list[float], list[float], float]:
        return self._robot.get_joint_state()

    def _read_raw_images(self) -> tuple[dict[str, np.ndarray], float]:
        if not self.enable_cameras or self._camera_rig is None:
            return {}, time.time()
        return self._camera_rig.get_images()

    @staticmethod
    def _encode_images(raw_images: dict[str, np.ndarray]) -> dict[str, bytes]:
        return {name: _encode_jpg(frame) for name, frame in raw_images.items() if frame is not None}

    def get_state_observation(self) -> dict:
        state, joint_velocity, state_ts = self._read_robot_state()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "state": state,
            "joint_velocity": joint_velocity,
            "action": state,
            "state_timestamp": state_ts,
            "timestamp": state_ts,
            "pending_actions": pending,
        }

    def get_image_observation(self) -> dict:
        raw_images, image_ts = self._read_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "images": self._encode_images(raw_images),
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def get_raw_image_observation(self) -> dict:
        raw_images, image_ts = self._read_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "raw_images": raw_images,
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def close(self) -> None:
        if self._camera_rig is not None:
            self._camera_rig.stop()
        self._cameras_started = False
        self._robot.disconnect()

    def drop_frame(self) -> None:
        if self._camera_rig is not None and self._cameras_started:
            self._camera_rig.drop_frame()

    def start(self) -> None:
        if self._camera_rig is not None and not self._cameras_started:
            self._camera_rig.start()
            self._cameras_started = True


def _load_airbot_rlt_env_config(path: str) -> _airbot_env.AirbotRLTEnvConfig:
    clean_path = str(path or "").strip()
    if clean_path:
        return _airbot_env.AirbotRLTEnvConfig.from_path(clean_path)
    return _airbot_env.AirbotRLTEnvConfig.from_env()


def _require_sdk_joint_velocity(arm: object, side: str) -> np.ndarray:
    get_joint_vel = getattr(arm, "get_joint_vel", None)
    if not callable(get_joint_vel):
        raise RuntimeError(f"{side} follower SDK object does not provide get_joint_vel()")
    velocity = np.asarray(get_joint_vel(), dtype=np.float32).reshape(-1)
    if velocity.size < 6:
        raise RuntimeError(f"{side} follower get_joint_vel() must return at least 6 values, got shape {velocity.shape}")
    return velocity[:6]
