import jax
import numpy as np

from openpi.models import pi0_config
from openpi.policies import airbot_policy
from openpi.policies import policy as _policy


def test_airbot_inputs_outputs_round_trip():
    transform = airbot_policy.AirbotInputs(action_dim=32)
    example = airbot_policy.make_airbot_example()
    example["actions"] = np.ones((10, 14), dtype=np.float32)

    inputs = transform(example)
    assert inputs["state"].shape == (32,)
    assert inputs["actions"].shape == (10, 32)
    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert inputs["image_mask"]["base_0_rgb"]
    assert isinstance(inputs["prompt"], str)

    outputs = airbot_policy.AirbotOutputs()({"actions": inputs["actions"]})
    assert outputs["actions"].shape == (10, 14)


def test_airbot_inputs_handles_missing_images():
    transform = airbot_policy.AirbotInputs(action_dim=32)
    inputs = transform(
        {
            "state": np.ones((14,), dtype=np.float32),
            "actions": np.ones((10, 14), dtype=np.float32),
            "prompt": "do something",
        }
    )

    assert inputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert not inputs["image_mask"]["base_0_rgb"]
    assert not inputs["image_mask"]["left_wrist_0_rgb"]
    assert not inputs["image_mask"]["right_wrist_0_rgb"]


def test_policy_prepare_observation_for_airbot():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        use_rlt=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)
    policy = _policy.Policy(
        model,
        transforms=[airbot_policy.AirbotInputs(action_dim=config.action_dim, model_type=config.model_type)],
    )

    inputs, observation = policy.prepare_observation(airbot_policy.make_airbot_example())
    assert inputs["state"].shape == (32,)
    assert observation.state.shape == (1, 32)
    assert observation.images["base_0_rgb"].shape == (1, 224, 224, 3)
