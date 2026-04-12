import numpy as np

from openpi.rlt.airbot_env import AirbotRLTEnv
from openpi.rlt.airbot_env import AirbotRLTEnvConfig
from openpi.rlt.airbot_env import ArmObservation
from openpi.rlt.airbot_env import OperatorFeedback
from openpi.rlt.airbot_env import _edge_triggered_binary_gripper_command


class _FakeArm:
    def __init__(self, reset_joint_pos: tuple[float, ...], *, gripper_mode: str = "continuous", gripper_max_length: float = 0.07) -> None:
        self._reset_joint_pos = np.asarray(reset_joint_pos[:6], dtype=np.float32)
        self._reset_gripper = float(reset_joint_pos[6])
        self._gripper_mode = gripper_mode
        self._gripper_max_length = gripper_max_length
        self._joint_pos = self._reset_joint_pos.copy()
        self._gripper = self._reset_gripper
        self.applied_actions: list[np.ndarray] = []

    def reset(self) -> None:
        self._joint_pos = self._reset_joint_pos.copy()
        self._gripper = self._reset_gripper

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        self.applied_actions.append(action)
        self._joint_pos = action[:6]
        if self._gripper_mode == "continuous":
            self._gripper = float(np.clip(action[6], 0.0, self._gripper_max_length))
        elif action[6] <= -0.5:
            self._gripper = 0.0
        elif action[6] >= 0.5:
            self._gripper = self._gripper_max_length
        return action

    def observe(self) -> ArmObservation:
        return ArmObservation(
            joint_pos=self._joint_pos.copy(),
            joint_vel=np.zeros((6,), dtype=np.float32),
            joint_torque=np.zeros((6,), dtype=np.float32),
            ee_pose=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gripper_pos=self._gripper,
        )

    def close(self) -> None:
        return None


class _FakeCamera:
    def __init__(self, value: int) -> None:
        self._frame = np.full((32, 48, 3), value, dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self._frame.copy()

    def close(self) -> None:
        return None


class _FakeOperator:
    def __init__(self, *, override: np.ndarray | None = None, feedbacks: list[OperatorFeedback] | None = None) -> None:
        self._override = None if override is None else np.asarray(override, dtype=np.float32).copy()
        self._feedbacks = list(feedbacks or [])
        self.prepared = False

    def prepare_for_episode(self) -> None:
        self.prepared = True

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        del current_state
        if self._override is None:
            return np.asarray(policy_action, dtype=np.float32).copy(), False
        return self._override.copy(), True

    def consume_feedback(self) -> OperatorFeedback:
        if self._feedbacks:
            return self._feedbacks.pop(0)
        return OperatorFeedback()

    def close(self) -> None:
        return None


def _make_env(*, operator: _FakeOperator) -> tuple[AirbotRLTEnv, _FakeArm, _FakeArm]:
    config = AirbotRLTEnvConfig(control_hz=10_000, display_images=False, fake_env=True, prompt="ethernet")
    left_arm = _FakeArm(
        config.left_arm.reset_joint_pos,
        gripper_mode=config.left_arm.gripper_mode,
        gripper_max_length=config.left_arm.gripper_max_length,
    )
    right_arm = _FakeArm(
        config.right_arm.reset_joint_pos,
        gripper_mode=config.right_arm.gripper_mode,
        gripper_max_length=config.right_arm.gripper_max_length,
    )
    cameras = {
        "cam_high": _FakeCamera(32),
        "cam_left_wrist": _FakeCamera(64),
        "cam_right_wrist": _FakeCamera(96),
    }
    env = AirbotRLTEnv(config, left_arm=left_arm, right_arm=right_arm, cameras=cameras, operator=operator)
    return env, left_arm, right_arm


def test_airbot_env_returns_manual_sparse_rewards() -> None:
    operator = _FakeOperator(feedbacks=[OperatorFeedback(), OperatorFeedback(reward=1.0)])
    env, _, _ = _make_env(operator=operator)

    reset_obs = env.reset()
    assert operator.prepared
    assert reset_obs["prompt"] == "ethernet"

    actions = np.stack(
        [
            np.arange(14, dtype=np.float32),
            np.arange(14, dtype=np.float32) + 1,
        ]
    )
    obs, reward, done, info = env.step(actions)

    np.testing.assert_allclose(info["step_rewards"], [0.0, 1.0])
    assert reward == 1.0
    assert not done
    assert info["success"] == 1.0
    np.testing.assert_allclose(obs["state"][:7], np.asarray([1, 2, 3, 4, 5, 6, 0.07], dtype=np.float32))
    np.testing.assert_allclose(obs["state"][7:14], np.asarray([8, 9, 10, 11, 12, 13, 0.07], dtype=np.float32))
    np.testing.assert_allclose(obs["proprio"][:14], obs["state"])
    assert obs["state"].shape == (14,)
    assert obs["proprio"].shape == (28,)
    env.close()


def test_airbot_env_uses_intervention_action() -> None:
    intervention = np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0])
    operator = _FakeOperator(
        override=intervention,
        feedbacks=[OperatorFeedback(terminate_episode=True, terminal_reason="operator_terminate")],
    )
    env, left_arm, right_arm = _make_env(operator=operator)

    env.reset()
    obs, reward, done, info = env.step(np.zeros((2, 14), dtype=np.float32))

    assert reward == 0.0
    assert done
    assert info["intervened"] == 1.0
    np.testing.assert_allclose(info["step_rewards"], [0.0])
    np.testing.assert_allclose(info["intervene_action"], intervention)
    np.testing.assert_allclose(left_arm.applied_actions[0], intervention[:7])
    np.testing.assert_allclose(right_arm.applied_actions[0], intervention[7:])
    np.testing.assert_allclose(obs["state"][:7], np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        obs["state"][7:14], np.asarray([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.07], dtype=np.float32)
    )
    np.testing.assert_allclose(obs["proprio"][:14], obs["state"])
    assert obs["state"].shape == (14,)
    assert obs["proprio"].shape == (28,)
    env.close()


def test_airbot_env_can_reward_and_terminate_same_step() -> None:
    operator = _FakeOperator(
        feedbacks=[OperatorFeedback(reward=1.0, terminate_episode=True, terminal_reason="success")]
    )
    env, _, _ = _make_env(operator=operator)

    env.reset()
    _, reward, done, info = env.step(np.zeros((1, 14), dtype=np.float32))

    assert reward == 1.0
    assert done
    assert info["terminal_reason"] == "success"
    np.testing.assert_allclose(info["step_rewards"], [1.0])
    env.close()


def test_edge_triggered_binary_gripper_command() -> None:
    command, closed = _edge_triggered_binary_gripper_command(0.06, None, threshold=0.035)
    assert command == 0.0
    assert not closed

    command, closed = _edge_triggered_binary_gripper_command(0.0, False, threshold=0.035)
    assert command == -1.0
    assert closed

    command, closed = _edge_triggered_binary_gripper_command(0.07, True, threshold=0.035)
    assert command == 1.0
    assert not closed
