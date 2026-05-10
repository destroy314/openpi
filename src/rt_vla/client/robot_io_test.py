from __future__ import annotations

import numpy as np

from openpi.rlt.airbot_env import CameraConfig
from rt_vla.client.robot_io import AirbotRealObserver
from rt_vla.client.robot_io import _RLTOpenCVCameraRig


class _FakeRobot:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_joint_state(self) -> tuple[list[float], list[float], float]:
        return list(np.arange(14, dtype=np.float32)), list(np.arange(14, dtype=np.float32) + 0.5), 123.0


def test_airbot_real_observer_exposes_sdk_joint_velocity() -> None:
    robot = _FakeRobot()
    observer = AirbotRealObserver(
        airbot_host="localhost",
        left_port=50051,
        right_port=50053,
        top_camera_id="cam_high",
        left_camera_id="cam_left_wrist",
        right_camera_id="cam_right_wrist",
        enable_cameras=False,
        image_size=(640, 480),
        fps=30,
        robot=robot,
    )

    state = observer.get_state_observation()

    assert robot.connected
    np.testing.assert_allclose(state["state"], np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(state["joint_velocity"], np.arange(14, dtype=np.float32) + 0.5)
    assert state["timestamp"] == 123.0

    observer.close()
    assert robot.disconnected


def test_rlt_opencv_camera_rig_uses_rlt_camera_names_and_outputs_rt_vla_image_keys() -> None:
    cameras = {
        "cam_high": CameraConfig(index=4, width=64, height=48, fake=True),
        "cam_left_wrist": CameraConfig(index=2, width=64, height=48, fake=True),
        "cam_right_wrist": CameraConfig(index=0, width=64, height=48, fake=True),
    }
    rig = _RLTOpenCVCameraRig(cameras)

    try:
        images, timestamp = rig.get_images()
    finally:
        rig.stop()

    assert set(images) == {"high", "left_hand", "right_hand"}
    assert timestamp > 0
    for image in images.values():
        assert image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
