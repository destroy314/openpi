import jax
import jax.numpy as jnp
import numpy as np
import pathlib
import pytest
import queue
import threading
from types import SimpleNamespace

from scripts import replay_rlt_training as replay_script
from scripts import train_rlt_online as online
from openpi.models import pi0_config
from openpi.rlt import checkpointing
from openpi.rlt import replay_buffer
from openpi.rlt import token_module
from openpi.rlt import trainer


def _make_transition_batch(batch: replay_buffer.TransitionBatch) -> replay_buffer.TransitionBatch:
    return replay_buffer.TransitionBatch(
        state=jnp.asarray(batch.state),
        action=jnp.asarray(batch.action),
        reference_action=jnp.asarray(batch.reference_action),
        reward=jnp.asarray(batch.reward),
        next_state=jnp.asarray(batch.next_state),
        next_reference_action=jnp.asarray(batch.next_reference_action),
        bootstrap_steps=jnp.asarray(batch.bootstrap_steps),
        done=jnp.asarray(batch.done),
    )


def test_replay_buffer_sample_shapes():
    buffer = replay_buffer.ReplayBuffer(capacity=8, state_dim=16, action_dim=32)
    for index in range(8):
        buffer.add(
            state=np.full((16,), index, dtype=np.float32),
            action=np.full((32,), index, dtype=np.float32),
            reference_action=np.full((32,), index + 1, dtype=np.float32),
            reward=float(index),
            next_state=np.full((16,), index + 2, dtype=np.float32),
            next_reference_action=np.full((32,), index + 3, dtype=np.float32),
            bootstrap_steps=4,
            done=False,
        )

    batch = buffer.sample(4, rng=np.random.default_rng(0))
    assert batch.state.shape == (4, 16)
    assert batch.action.shape == (4, 32)
    assert batch.next_reference_action.shape == (4, 32)
    assert batch.bootstrap_steps.shape == (4,)


def test_masked_reconstruction_loss_averages_over_embedding_dim():
    reconstruction = jnp.ones((2, 3, 4), dtype=jnp.float32)
    target = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    mask = jnp.array([[True, True, False], [True, False, False]])

    loss = token_module.masked_reconstruction_loss(reconstruction, target, mask)

    np.testing.assert_allclose(loss, np.ones((2,), dtype=np.float32))


def test_rlt_training_step_and_checkpoint(tmp_path):
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        use_rlt=True,
        rlt_actor_enabled=True,
        action_dim=14,
        action_horizon=50,
        rlt_action_horizon=10,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    state_dim = config.rlt_token_dim + config.action_dim
    action_dim = config.rlt_env_action_dim * config.rlt_action_horizon
    buffer = replay_buffer.ReplayBuffer(capacity=32, state_dim=state_dim, action_dim=action_dim)
    for index in range(16):
        value = float(index)
        buffer.add(
            state=np.full((state_dim,), value, dtype=np.float32),
            action=np.full((action_dim,), value, dtype=np.float32),
            reference_action=np.full((action_dim,), value + 1, dtype=np.float32),
            reward=value,
            next_state=np.full((state_dim,), value + 2, dtype=np.float32),
            next_reference_action=np.full((action_dim,), value + 3, dtype=np.float32),
            bootstrap_steps=config.rlt_action_horizon,
            done=index % 3 == 0,
        )

    actor_model, actor_state = trainer.init_actor_state(
        jax.random.key(4),
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.rlt_actor_hidden_dim,
        learning_rate=1e-3,
    )
    actor_state = trainer.apply_actor_params(
        actor_state,
        {"step": 0, "actor_params": trainer.extract_actor_params(model)},
    )
    critic_model, critic_state = trainer.init_critic_state(
        jax.random.key(1),
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        learning_rate=1e-3,
    )
    batch = _make_transition_batch(buffer.sample(8, rng=np.random.default_rng(1)))
    online_config = trainer.OnlineRLTConfig(reference_dropout=1.0)

    critic_state, critic_info = trainer.critic_step_with_actor(
        actor_model,
        actor_state,
        critic_model,
        critic_state,
        batch,
        jax.random.key(3),
        online_config,
    )
    assert critic_state.step == 1
    assert "critic_loss" in critic_info

    actor_state, actor_info = trainer.actor_step_with_actor(
        actor_model,
        critic_model,
        critic_state,
        actor_state,
        batch,
        jax.random.key(2),
        online_config,
    )
    assert int(actor_state.step) == 1
    assert "actor_loss" in actor_info

    step_dir = checkpointing.save_checkpoint(
        tmp_path,
        step=7,
        policy_state=trainer.bundle_actor_train_state(actor_state),
        critic_state=trainer.bundle_critic_state(critic_state),
        norm_stats=None,
        asset_id=None,
    )
    restored_policy = checkpointing.restore_bundle(step_dir, "policy_state")
    restored_critic = checkpointing.restore_bundle(step_dir, "critic_state")
    restored_actor = trainer.restore_actor_state(actor_model, restored_policy, learning_rate=1e-3)
    assert int(restored_actor.step) == 1
    assert int(restored_critic["step"]) == 1


def test_patch_chunk_with_executed_prefix_replaces_action_and_intervention_reference():
    chunk = online.PlannedChunk(
        step=0,
        state=np.zeros((4,), dtype=np.float32),
        action=np.arange(12, dtype=np.float32),
        reference_action=(100 + np.arange(12, dtype=np.float32)),
    )
    executed_actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    intervened_mask = np.asarray([False, True], dtype=bool)

    online._patch_chunk_with_executed_prefix(
        chunk,
        executed_actions=executed_actions,
        intervened_mask=intervened_mask,
        action_horizon=3,
        action_dim=4,
        env_action_dim=2,
    )

    patched_action = chunk.action.reshape(3, 4)
    patched_reference = chunk.reference_action.reshape(3, 4)

    np.testing.assert_allclose(patched_action[0, :2], [1.0, 2.0])
    np.testing.assert_allclose(patched_action[1, :2], [3.0, 4.0])
    np.testing.assert_allclose(patched_reference[0, :2], [100.0, 101.0])
    np.testing.assert_allclose(patched_reference[1, :2], [3.0, 4.0])


def test_resolve_reference_chunk_uses_single_cached_vla_plan():
    reference_plan_history = {
        0: np.arange(10, dtype=np.float32).reshape(10, 1),
        6: (100 + np.arange(10, dtype=np.float32)).reshape(10, 1),
    }

    reference_chunk = online._resolve_reference_chunk(
        reference_plan_history,
        step=6,
        action_horizon=4,
        replan_origin_step=0,
        vla_replan_horizon=6,
    )

    np.testing.assert_allclose(reference_chunk, [100.0, 101.0, 102.0, 103.0])


def test_resolve_reference_chunk_extends_latest_plan_when_next_replan_missing():
    reference_plan_history = {
        0: np.arange(10, dtype=np.float32).reshape(10, 1),
    }

    reference_chunk = online._resolve_reference_chunk(
        reference_plan_history,
        step=6,
        action_horizon=3,
        replan_origin_step=0,
        vla_replan_horizon=6,
    )

    np.testing.assert_allclose(reference_chunk, [6.0, 7.0, 8.0])


def test_resolve_reference_chunk_raises_when_chunk_would_cross_plan_end():
    reference_plan_history = {
        0: np.arange(10, dtype=np.float32).reshape(10, 1),
    }

    with pytest.raises(ValueError, match="cannot provide a contiguous chunk"):
        online._resolve_reference_chunk(
            reference_plan_history,
            step=8,
            action_horizon=4,
            replan_origin_step=0,
            vla_replan_horizon=6,
        )


def test_queue_transition_records_before_enqueue():
    class Recorder:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

    sample_queue: queue.Queue = queue.Queue()
    recorder = Recorder()
    item = online.ReplayItem(
        state=np.asarray([1.0, 2.0], dtype=np.float32),
        action=np.asarray([3.0, 4.0], dtype=np.float32),
        reference_action=np.asarray([5.0, 6.0], dtype=np.float32),
        reward=1.0,
        next_state=np.asarray([7.0, 8.0], dtype=np.float32),
        next_reference_action=np.asarray([9.0, 10.0], dtype=np.float32),
        bootstrap_steps=1,
        done=False,
        env_step=2,
    )

    online._queue_transition(sample_queue, item, transition_recorder=recorder)

    assert recorder.items == [item]
    assert sample_queue.get_nowait() == item


def test_async_learner_queue_policy_update_and_checkpoint(monkeypatch, tmp_path):
    class FakePolicyState:
        def __init__(self, *, step: int, actor_value: float):
            self.step = step
            self.actor_value = actor_value

    class FakeCriticState:
        def __init__(self, *, step: int):
            self.step = step

    saved_checkpoints: list[tuple[int, dict, dict]] = []

    def fake_get_config(_name):
        return SimpleNamespace(
            model=SimpleNamespace(rlt_actor_hidden_dim=8),
            data=SimpleNamespace(
                create=lambda *_args: SimpleNamespace(asset_id=None, use_quantile_norm=False),
            ),
            assets_dirs=(),
        )

    def fake_create_trained_policy(_config, _checkpoint_dir):
        return SimpleNamespace(model=object())

    def fake_init_actor_state(_rng, state_dim, action_dim, *, hidden_dim, learning_rate):
        del state_dim, action_dim, hidden_dim, learning_rate
        return object(), FakePolicyState(step=0, actor_value=0.0)

    def fake_init_critic_state(_rng, state_dim, action_dim, *, hidden_dim, learning_rate):
        del state_dim, action_dim, hidden_dim, learning_rate
        return object(), FakeCriticState(step=0)

    def fake_critic_step_with_actor(_actor_model, _actor_state, _critic_model, critic_state, _batch, _rng, _config):
        return FakeCriticState(step=critic_state.step + 1), {"critic_loss": jnp.asarray(1.0)}

    def fake_actor_step_with_actor(_actor_model, _critic_model, _critic_state, policy_state, _batch, _rng, _config):
        return (
            FakePolicyState(step=policy_state.step + 1, actor_value=policy_state.actor_value + 1.0),
            {"actor_loss": jnp.asarray(0.5)},
        )

    def fake_bundle_actor_state(state):
        return {"step": state.step, "actor_value": state.actor_value}

    def fake_apply_actor_params(state, bundle):
        actor_payload = bundle.get("actor_params", bundle)
        return FakePolicyState(step=int(bundle["step"]), actor_value=float(actor_payload["actor_value"]))

    def fake_bundle_actor_train_state(state):
        return {"step": state.step, "actor_value": state.actor_value}

    def fake_restore_actor_state(_actor_model, bundle, _learning_rate):
        return FakePolicyState(step=int(bundle["step"]), actor_value=float(bundle["actor_value"]))

    def fake_bundle_critic_state(state):
        return {"step": state.step}

    def fake_save_checkpoint(checkpoint_dir, step, *, policy_state, critic_state, norm_stats, asset_id):
        del checkpoint_dir, norm_stats, asset_id
        saved_checkpoints.append((step, policy_state, critic_state))
        return pathlib.Path(tmp_path) / str(step)

    def fake_save_initial_state(_repro_dir, _item, _bundle):
        return pathlib.Path(tmp_path)

    def fake_save_stop_signal(_repro_dir, _final_step):
        return pathlib.Path(tmp_path) / "stop_signal.json"

    signal_calls: list[tuple[object, object]] = []

    def fake_signal(sig, handler):
        signal_calls.append((sig, handler))

    monkeypatch.setattr(online._config, "get_config", fake_get_config)
    monkeypatch.setattr(online._policy_config, "create_trained_policy", fake_create_trained_policy)
    monkeypatch.setattr(online._trainer, "init_actor_state", fake_init_actor_state)
    monkeypatch.setattr(online._trainer, "init_critic_state", fake_init_critic_state)
    monkeypatch.setattr(online._trainer, "critic_step_with_actor", fake_critic_step_with_actor)
    monkeypatch.setattr(online._trainer, "actor_step_with_actor", fake_actor_step_with_actor)
    monkeypatch.setattr(online._trainer, "bundle_actor_params", fake_bundle_actor_state)
    monkeypatch.setattr(online._trainer, "apply_actor_params", fake_apply_actor_params)
    monkeypatch.setattr(online._trainer, "bundle_actor_train_state", fake_bundle_actor_train_state)
    monkeypatch.setattr(online._trainer, "restore_actor_state", fake_restore_actor_state)
    monkeypatch.setattr(online._trainer, "bundle_critic_state", fake_bundle_critic_state)
    monkeypatch.setattr(online._checkpointing, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(online._repro, "save_initial_state", fake_save_initial_state)
    monkeypatch.setattr(online._repro, "save_stop_signal", fake_save_stop_signal)
    monkeypatch.setattr(online.signal, "signal", fake_signal)

    args = online.Args(
        batch_size=1,
        warmup_steps=0,
        utd_ratio=1,
        save_interval=2,
        log_interval=1000,
        overwrite=True,
        checkpoint_dir=tmp_path,
        init_checkpoint_dir=tmp_path,
        online=trainer.OnlineRLTConfig(actor_update_interval=1),
    )

    sample_queue: queue.Queue = queue.Queue()
    policy_queue: queue.Queue = queue.Queue(maxsize=1)
    status_queue: queue.Queue = queue.Queue()

    learner_thread = threading.Thread(
        target=online._learner_main,
        args=(args, tmp_path, 3, 2, 8, sample_queue, policy_queue, status_queue),
        daemon=True,
    )
    learner_thread.start()

    init_message = online._wait_for_learner_init(status_queue)
    assert init_message.start_step == 0
    local_policy_state = FakePolicyState(step=0, actor_value=0.0)

    sample_queue.put(
        online.ReplayItem(
            state=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            action=np.asarray([4.0, 5.0], dtype=np.float32),
            reference_action=np.asarray([6.0, 7.0], dtype=np.float32),
            reward=1.0,
            next_state=np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
            next_reference_action=np.asarray([11.0, 12.0], dtype=np.float32),
            bootstrap_steps=1,
            done=False,
            env_step=2,
        )
    )

    policy_update = policy_queue.get(timeout=2.0)
    policy_queue.put(policy_update)
    local_policy_state = online._drain_policy_updates(policy_queue, local_policy_state)

    assert local_policy_state.step == 1
    assert local_policy_state.actor_value == 1.0

    sample_queue.put(online.StopSignal(final_step=3))
    learner_thread.join(timeout=2.0)
    assert not learner_thread.is_alive()

    online._check_learner_status(status_queue)
    assert [step for step, _, _ in saved_checkpoints] == [2, 3]
    assert saved_checkpoints[0][1]["actor_value"] == 1.0
    assert saved_checkpoints[1][0] == 3
    assert signal_calls[0] == (online.signal.SIGINT, online.signal.SIG_IGN)


def test_replay_script_replays_recorded_transitions(monkeypatch, tmp_path):
    class FakePolicyState:
        def __init__(self, *, step: int, actor_value: float):
            self.step = step
            self.actor_value = actor_value

    class FakeCriticState:
        def __init__(self, *, step: int):
            self.step = step

    saved_checkpoints: list[tuple[int, dict, dict]] = []

    def fake_load_manifest(_repro_dir):
        return {
            "args": {
                "config": "fake_config",
                "init_checkpoint_dir": str(tmp_path / "stage1"),
                "checkpoint_dir": str(tmp_path / "ignored"),
                "max_env_steps": 10,
                "warmup_steps": 0,
                "replay_capacity": 8,
                "batch_size": 1,
                "utd_ratio": 1,
                "diffusion_steps": 10,
                "chunk_stride": 2,
                "actor_lr": 1e-3,
                "critic_lr": 1e-3,
                "save_interval": 2,
                "log_interval": 1000,
                "overwrite": False,
                "resume": False,
                "seed": 0,
                "record_repro": True,
                "repro_shard_size": 8,
                "online": {
                    "discount": 0.99,
                    "actor_bc_weight": 1.0,
                    "reference_dropout": 0.5,
                    "actor_update_interval": 1,
                    "target_tau": 0.005,
                },
            },
            "state_dim": 3,
            "action_dim": 2,
            "actor_hidden_dim": 8,
            "asset_id": None,
            "start_step": 0,
        }

    def fake_get_config(_name):
        return SimpleNamespace(model=SimpleNamespace(rlt_actor_hidden_dim=8))

    def fake_init_actor_state(_rng, state_dim, action_dim, *, hidden_dim, learning_rate):
        del state_dim, action_dim, hidden_dim, learning_rate
        return object(), FakePolicyState(step=0, actor_value=0.0)

    def fake_init_critic_state(_rng, state_dim, action_dim, *, hidden_dim, learning_rate):
        del state_dim, action_dim, hidden_dim, learning_rate
        return object(), FakeCriticState(step=0)

    def fake_restore_actor_state(_actor_model, bundle, _learning_rate):
        return FakePolicyState(step=int(bundle["step"]), actor_value=float(bundle["actor_value"]))

    def fake_restore_critic_state(_critic_model, bundle, _learning_rate):
        return FakeCriticState(step=int(bundle["step"]))

    def fake_critic_step_with_actor(_actor_model, _actor_state, _critic_model, critic_state, _batch, _rng, _config):
        return FakeCriticState(step=critic_state.step + 1), {"critic_loss": jnp.asarray(1.0)}

    def fake_actor_step_with_actor(_actor_model, _critic_model, _critic_state, actor_state, _batch, _rng, _config):
        return (
            FakePolicyState(step=actor_state.step + 1, actor_value=actor_state.actor_value + 1.0),
            {"actor_loss": jnp.asarray(0.5)},
        )

    def fake_bundle_actor_train_state(state):
        return {"step": state.step, "actor_value": state.actor_value}

    def fake_bundle_critic_state(state):
        return {"step": state.step}

    def fake_save_checkpoint(checkpoint_dir, step, *, policy_state, critic_state, norm_stats, asset_id):
        del checkpoint_dir, norm_stats, asset_id
        saved_checkpoints.append((step, policy_state, critic_state))
        return pathlib.Path(tmp_path) / str(step)

    monkeypatch.setattr(replay_script._repro, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(replay_script._config, "get_config", fake_get_config)
    monkeypatch.setattr(replay_script._trainer, "init_actor_state", fake_init_actor_state)
    monkeypatch.setattr(replay_script._trainer, "init_critic_state", fake_init_critic_state)
    monkeypatch.setattr(replay_script._trainer, "restore_actor_state", fake_restore_actor_state)
    monkeypatch.setattr(replay_script._trainer, "restore_critic_state", fake_restore_critic_state)
    monkeypatch.setattr(replay_script._trainer, "critic_step_with_actor", fake_critic_step_with_actor)
    monkeypatch.setattr(replay_script._trainer, "actor_step_with_actor", fake_actor_step_with_actor)
    monkeypatch.setattr(replay_script._trainer, "bundle_actor_train_state", fake_bundle_actor_train_state)
    monkeypatch.setattr(replay_script._trainer, "bundle_critic_state", fake_bundle_critic_state)
    monkeypatch.setattr(replay_script._repro, "load_initial_state", lambda _repro_dir, item: {"step": 0, "actor_value": 0.0} if item == "policy_state" else {"step": 0})
    monkeypatch.setattr(
        replay_script._repro,
        "iter_transition_dicts",
        lambda _repro_dir: iter(
            [
                {
                    "state": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    "action": np.asarray([4.0, 5.0], dtype=np.float32),
                    "reference_action": np.asarray([6.0, 7.0], dtype=np.float32),
                    "reward": 1.0,
                    "next_state": np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
                    "next_reference_action": np.asarray([11.0, 12.0], dtype=np.float32),
                    "bootstrap_steps": 1,
                    "done": False,
                    "env_step": 2,
                }
            ]
        ),
    )
    monkeypatch.setattr(replay_script._repro, "load_stop_signal", lambda _repro_dir: {"final_step": 3})
    monkeypatch.setattr(replay_script._checkpointing, "save_checkpoint", fake_save_checkpoint)

    replay_script.main(
        replay_script.Args(
            repro_dir=tmp_path / "repro",
            checkpoint_dir=tmp_path / "replayed",
            overwrite=True,
        )
    )

    assert [step for step, _, _ in saved_checkpoints] == [2, 3]
    assert saved_checkpoints[-1][1]["actor_value"] == 1.0
