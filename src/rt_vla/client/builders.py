from __future__ import annotations

from .config import Config
from .feedback import KeyboardFeedbackConfig
from .feedback import KeyboardFeedbackProvider
from .feedback import LeaderArmActionOverride
from .feedback import NoopFeedbackProvider
from .executor import BaseExecutor
from .executor import OnDeviceMpcExecutor
from .executor import RawActionExecutor
from .rlt_actor_runtime import RLTFeatureClient
from .rlt_local_client import RLTLocalClient
from .rlt_training_runtime import RLTTrainingRuntime
from .robot_io import AirbotActuator
from .robot_io import AirbotRealObserver
from .robot_io import BaseActuator
from .robot_io import BaseObserver
from .robot_io import MockStateObserver
from .robot_io import NoopActuator
from .robot_io import create_airbot_robot


def _build_shared_airbot_robot(cfg: Config):
    if cfg.observer.name != "airbot_real":
        return None
    obs = cfg.observer
    ex = cfg.executor
    if (
        obs.airbot_host != ex.airbot_host
        or int(obs.airbot_left_port) != int(ex.airbot_left_port)
        or int(obs.airbot_right_port) != int(ex.airbot_right_port)
    ):
        raise ValueError("observer/executor airbot endpoint mismatch; MPC runtime requires a shared robot connection")
    robot = create_airbot_robot(
        airbot_host=obs.airbot_host,
        left_port=int(obs.airbot_left_port),
        right_port=int(obs.airbot_right_port),
    )
    robot.connect()
    return robot


def build_observer(cfg: Config, pending_actions_provider, shared_robot=None) -> BaseObserver:
    observer_map = {
        "mock": MockStateObserver,
        "airbot_real": AirbotRealObserver,
    }
    observer_cls = observer_map.get(cfg.observer.name)
    if observer_cls is None:
        raise ValueError(f"Unsupported observer.name={cfg.observer.name!r}")
    if observer_cls is AirbotRealObserver:
        return observer_cls.from_config(cfg, pending_actions_provider, robot=shared_robot)
    return observer_cls.from_config(cfg, pending_actions_provider)


def _build_actuator(cfg: Config, shared_robot=None) -> BaseActuator:
    observer_cls = {
        "mock": NoopActuator,
        "airbot_real": AirbotActuator,
    }
    observer_cls = observer_cls.get(cfg.observer.name)
    if observer_cls is None:
        raise ValueError(f"Unsupported observer.name={cfg.observer.name!r}")
    if observer_cls is AirbotActuator:
        return observer_cls.from_config(cfg, robot=shared_robot)
    return observer_cls.from_config(cfg)


def build_executor(cfg: Config, shared_robot=None) -> BaseExecutor:
    executor_map = {
        "raw_action": RawActionExecutor,
        "ondevice_mpc": OnDeviceMpcExecutor,
    }
    executor_cls = executor_map.get(cfg.executor.name)
    if executor_cls is None:
        raise ValueError(f"Unsupported executor.name={cfg.executor.name!r}")
    actuator = _build_actuator(cfg, shared_robot=shared_robot)
    return executor_cls.from_config(cfg, actuator)


def build_runtime_components(cfg: Config, pending_actions_provider):
    shared_robot = _build_shared_airbot_robot(cfg)
    executor = build_executor(cfg, shared_robot=shared_robot)
    observer = build_observer(cfg, pending_actions_provider, shared_robot=shared_robot)
    return executor, observer


def build_feedback_provider(cfg: Config):
    if cfg.feedback.name == "noop":
        if cfg.feedback.enable_leader_override:
            raise ValueError("feedback.enable_leader_override requires feedback.name='keyboard'")
        return NoopFeedbackProvider()
    if cfg.feedback.name == "keyboard":
        action_override_callback = None
        if cfg.feedback.enable_leader_override:
            action_override_callback = LeaderArmActionOverride(
                left_leader_port=cfg.feedback.left_leader_port,
                right_leader_port=cfg.feedback.right_leader_port,
            )
        return KeyboardFeedbackProvider(
            config=KeyboardFeedbackConfig(
                intervention_toggle_key=cfg.feedback.intervention_toggle_key,
                reward_key=cfg.feedback.reward_key,
                terminate_episode_key=cfg.feedback.terminate_episode_key,
                abort_key=cfg.feedback.abort_key,
                key_debounce_sec=cfg.feedback.key_debounce_sec,
            ),
            action_override_callback=action_override_callback,
        )
    raise ValueError(f"Unsupported feedback.name={cfg.feedback.name!r}")


def build_rlt_local_client(cfg: Config, feature_transport, observer) -> RLTLocalClient:
    feature_client = RLTFeatureClient(
        transport=feature_transport,
        infer_endpoint=cfg.rlt.service.infer_endpoint,
        token_endpoint=cfg.rlt.service.token_endpoint,
        status_endpoint=cfg.rlt.service.status_endpoint,
    )
    return RLTLocalClient(
        config=cfg,
        feature_client=feature_client,
        observer=observer,
    )


def build_rlt_training_runtime(
    cfg: Config,
    *,
    feature_transport,
    actor_model,
    actor_state,
    learner_runtime,
    norm_stats,
    use_quantiles: bool,
    use_delta_joint_actions: bool,
    action_horizon: int,
    control_dt_s: float,
    apply_actor_params,
) -> RLTTrainingRuntime:
    executor, observer = build_runtime_components(cfg, pending_actions_provider=lambda: 0)
    feedback_provider = build_feedback_provider(cfg)
    local_client = build_rlt_local_client(cfg, feature_transport, observer)
    return RLTTrainingRuntime(
        config=cfg,
        local_client=local_client,
        actor_model=actor_model,
        actor_state=actor_state,
        executor=executor,
        learner_runtime=learner_runtime,
        feedback_provider=feedback_provider,
        norm_stats=norm_stats,
        use_quantiles=use_quantiles,
        use_delta_joint_actions=use_delta_joint_actions,
        action_horizon=action_horizon,
        control_dt_s=control_dt_s,
        apply_actor_params=apply_actor_params,
        prompt=cfg.rlt.observation.prompt,
    )
