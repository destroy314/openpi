"""Pure VLA rollout client for Airbot dual-arm robots.

Wraps ``AirbotRLTEnv`` (hardware I/O, camera streaming, rate control) and
connects to a running openpi policy server via WebSocket.  No leader-arm
intervention or keyboard dependency is used; the policy drives the robot
autonomously.

Usage
-----
    # minimal
    python -m examples.airbot.vla_rollout_client --host 192.168.1.10 --port 8000

    # load a custom env config file
    python -m examples.airbot.vla_rollout_client \\
        --host 192.168.1.10 \\
        --config-path /path/to/airbot_config.toml
"""

import dataclasses
import logging
import time

import einops
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

from openpi.rlt.airbot_env import (
    AirbotRLTEnv,
    AirbotRLTEnvConfig,
    ArmConfig,
    CameraConfig,
)


LOGGER = logging.getLogger(__name__)

# Init joint positions: mean of episode-initial states across all 34 episodes
# in icrlab/block_handover (frame_index == 0).  Units: radians / metres.
# fmt: off
_DEFAULT_LEFT_RESET  = (-0.3736, -0.8108,  0.6645,  1.4765, -0.8911, -1.3767,  0.0)
_DEFAULT_RIGHT_RESET = ( 0.3809, -0.8669,  0.7577, -1.5047,  0.9221,  1.6538,  0.0)
# fmt: on



@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    """IP address of the openpi policy server."""
    port: int = 8000

    action_horizon: int = 25
    """Number of actions to execute per inference call."""
    max_episodes: int = 100
    max_steps: int = 1_000_000
    """Max waypoint steps per episode (safety limit)."""

    prompt: str = "do some thing"
    """Language prompt sent to the policy server."""

    config_path: str | None = None
    """Path to a .toml or .json AirbotRLTEnvConfig file.
    When set, hardware/camera fields below are ignored."""
    control_hz: int = 25
    """Control frequency sent to AirbotRLTEnvConfig (ignored with --config-path)."""
    left_follower_port: int = 50051
    right_follower_port: int = 50053
    cam_high: int = 6
    cam_left_wrist: int = 3
    cam_right_wrist: int = 0
    display_images: bool = True

    arm_step: list[float] = dataclasses.field(default_factory=lambda: [1.0] * 7)
    """Per-joint max step size for interpolation (radians, last entry = gripper).
    Decrease for smoother, safer motion; default [1.0]*7 disables interpolation."""
    pre_infer_sleep: float = 1.0
    """Seconds to wait after the last waypoint before reading state for inference.
    Gives the arm time to settle at the target pose."""
    post_reset_sleep: float = 0.0
    """Seconds to wait after reset before starting the episode."""


def parse_obs(raw_obs: dict, prompt: str) -> dict:
    """Convert AirbotRLTEnv observation to the dict expected by the policy server.

    AirbotRLTEnv returns images as HWC uint8 RGB; the server expects CHW.
    Gripper values are kept in raw metres (icrlab/block_handover pipeline).
    """
    images: dict[str, np.ndarray] = {
        name: einops.rearrange(np.asarray(img, dtype=np.uint8), "h w c -> c h w")
        for name, img in raw_obs["images"].items()
    }
    return {
        "state": raw_obs["state"].copy().astype(np.float32),
        "images": images,
        "prompt": prompt,
    }


def parse_actions(raw: np.ndarray) -> np.ndarray:
    """Cast policy-server actions to float32.  Gripper is kept in raw metres."""
    return np.asarray(raw, dtype=np.float32)  # (T, 14)


def interpolate_action(
    arm_step: list[float],
    prev: np.ndarray,
    cur: np.ndarray,
) -> np.ndarray:
    """Return interpolated waypoints from prev (exclusive) to cur (inclusive).

    With the default arm_step=[1.0]*7 this returns [cur] unchanged.
    Decrease arm_step values to add intermediate waypoints for safer motion.
    """
    # Duplicate for dual arm: (7,) → (14,)
    steps = np.concatenate([arm_step, arm_step]).astype(np.float32)
    diff = np.abs(cur - prev)
    n = int(np.max(np.ceil(diff / steps)))
    if n <= 1:
        return cur[np.newaxis, :]
    return np.linspace(prev, cur, n + 1, dtype=np.float32)[1:]  # exclude start


def _build_env_config(args: Args) -> AirbotRLTEnvConfig:
    if args.config_path is not None:
        cfg = AirbotRLTEnvConfig.from_path(args.config_path)
        return dataclasses.replace(
            cfg,
            enable_intervention=False,
            prompt=args.prompt,
            max_episode_steps=args.max_steps,
            post_reset_sleep_sec=args.post_reset_sleep,
        )
    return AirbotRLTEnvConfig(
        prompt=args.prompt,
        control_hz=args.control_hz,
        max_episode_steps=args.max_steps,
        display_images=args.display_images,
        enable_intervention=False,
        post_reset_sleep_sec=args.post_reset_sleep,
        left_arm=ArmConfig(
            name="left",
            follower_port=args.left_follower_port,
            reset_joint_pos=_DEFAULT_LEFT_RESET,
        ),
        right_arm=ArmConfig(
            name="right",
            follower_port=args.right_follower_port,
            reset_joint_pos=_DEFAULT_RIGHT_RESET,
        ),
        cameras={
            "cam_high": CameraConfig(index=args.cam_high),
            "cam_left_wrist": CameraConfig(index=args.cam_left_wrist),
            "cam_right_wrist": CameraConfig(index=args.cam_right_wrist),
        },
    )


def main(args: Args) -> None:
    env_cfg = _build_env_config(args)
    env = AirbotRLTEnv(env_cfg)

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )
    LOGGER.info("Server metadata: %s", policy.get_server_metadata())

    # Warm-up: reset once and run a dummy inference to establish the WebSocket
    # connection and trigger any server-side JIT compilation.
    raw_obs = env.reset()
    policy.infer(parse_obs(raw_obs, args.prompt))
    LOGGER.info("Warm-up done.")

    try:
        for ep_idx in range(args.max_episodes):
            input(f"\n[Episode {ep_idx + 1}/{args.max_episodes}] Press Enter to start.")

            raw_obs = env.reset()

            # pre_action tracks the last commanded position for interpolation.
            # Layout: [left_joints×6, left_gripper, right_joints×6, right_gripper]
            pre_action: np.ndarray = raw_obs["state"].copy().astype(np.float32)

            action_buffer: np.ndarray | None = None
            buf_idx: int = args.action_horizon  # trigger inference immediately
            done = False
            waypoint_count = 0

            for _ in range(args.max_steps):
                # ---- re-query the policy server every action_horizon steps ----
                if action_buffer is None or buf_idx >= len(action_buffer):
                    buf_idx = 0
                    if args.pre_infer_sleep > 0:
                        time.sleep(args.pre_infer_sleep)

                    t0 = time.perf_counter()
                    result = policy.infer(parse_obs(raw_obs, args.prompt))
                    action_buffer = parse_actions(result["actions"])
                    LOGGER.info(
                        "Infer %.2fs | prompt: %r | ep_step: %d",
                        time.perf_counter() - t0,
                        args.prompt,
                        waypoint_count,
                    )

                action = action_buffer[buf_idx]  # (14,)
                buf_idx += 1

                # ---- interpolation → fine-grained waypoints ------------------
                waypoints = interpolate_action(args.arm_step, pre_action, action)
                pre_action = action.copy()

                for wp in waypoints:
                    # env.step() handles rate limiting (1/control_hz) internally.
                    raw_obs, _reward, done, _info = env.step(wp[np.newaxis, :])
                    waypoint_count += 1
                    if done:
                        break

                if done:
                    LOGGER.info("Episode %d done at step %d.", ep_idx + 1, waypoint_count)
                    break

    finally:
        env.close()
        LOGGER.info("Environment closed. Bye.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass  # Ctrl-C; env.close() in finally handles arm power-off
