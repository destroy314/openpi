from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from openpi.rlt import trainer
import rt_vla.client.rlt_deploy as deploy_module
from rt_vla.client.rlt_deploy import DeployActorConfig
from rt_vla.client.rlt_deploy import load_deploy_actor
from rt_vla.client.rlt_deploy import sample_deploy_action


def test_deploy_actor_loads_stage2_checkpoint_and_returns_action_chunk(monkeypatch, tmp_path: Path) -> None:
    actor_model, actor_state = trainer.init_actor_state(
        jax.random.key(0),
        state_dim=284,
        action_dim=28,
        hidden_dim=8,
        learning_rate=1e-3,
    )
    del actor_model
    actor_bundle = trainer.bundle_actor_train_state(actor_state)
    fake_train_config = SimpleNamespace(model=SimpleNamespace(rlt_actor_hidden_dim=8))

    monkeypatch.setattr(deploy_module._config, "get_config", lambda _name: fake_train_config)
    monkeypatch.setattr(deploy_module._checkpointing, "restore_bundle", lambda _path, _item: actor_bundle)

    deploy_actor = load_deploy_actor(
        DeployActorConfig(
            config_name="fake-rlt",
            checkpoint_dir=tmp_path / "4",
            action_horizon=2,
            actor_lr=1e-3,
        )
    )
    action = sample_deploy_action(
        deploy_actor,
        rlt_state=np.zeros((284,), dtype=np.float32),
        reference_action=np.zeros((2, 14), dtype=np.float32),
        deterministic=True,
    )

    assert action.shape == (2, 14)
    assert np.isfinite(action).all()
