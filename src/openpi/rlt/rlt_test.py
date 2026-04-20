import jax
import jax.numpy as jnp
import numpy as np

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

    policy_state = trainer.init_policy_state(model, learning_rate=1e-3)
    critic_model, critic_state = trainer.init_critic_state(
        jax.random.key(1),
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        learning_rate=1e-3,
    )
    batch = _make_transition_batch(buffer.sample(8, rng=np.random.default_rng(1)))
    online_config = trainer.OnlineRLTConfig(reference_dropout=1.0)

    critic_state, critic_info = trainer.critic_step(
        critic_model,
        critic_state,
        policy_state,
        batch,
        jax.random.key(3),
        online_config,
    )
    assert critic_state.step == 1
    assert "critic_loss" in critic_info

    policy_state, actor_info = trainer.actor_step(
        critic_model,
        critic_state,
        policy_state,
        batch,
        jax.random.key(2),
        online_config,
    )
    assert int(policy_state.step) == 1
    assert "actor_loss" in actor_info

    step_dir = checkpointing.save_checkpoint(
        tmp_path,
        step=7,
        params=policy_state.params.to_pure_dict(),
        policy_state=trainer.bundle_policy_state(policy_state),
        critic_state=trainer.bundle_critic_state(critic_state),
        norm_stats=None,
        asset_id=None,
    )
    restored_policy = checkpointing.restore_bundle(step_dir, "policy_state")
    restored_critic = checkpointing.restore_bundle(step_dir, "critic_state")
    assert int(restored_policy["step"]) == 1
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
