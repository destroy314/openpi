import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    output_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    zero_output_init: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        if self.zero_output_init:
            return nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.normal(0.01),
                bias_init=nn.initializers.zeros,
            )(x)
        return nn.Dense(self.output_dim)(x)


class GaussianActor(nn.Module):
    """Reference-conditioned Gaussian actor for chunk refinement."""

    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    init_std: float = 0.05

    @nn.compact
    def __call__(self, state: jax.Array, reference_action: jax.Array) -> tuple[jax.Array, jax.Array]:
        features = jnp.concatenate([state, reference_action], axis=-1)
        delta = MLP(self.action_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, zero_output_init=True)(features)
        mean = reference_action + delta
        std = jnp.full(mean.shape, self.init_std, dtype=mean.dtype)
        return mean, std


class Critic(nn.Module):
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        inputs = jnp.concatenate([state, action], axis=-1)
        value = MLP(1, hidden_dim=self.hidden_dim, num_layers=self.num_layers)(inputs)
        return jnp.squeeze(value, axis=-1)


class TwinCritic(nn.Module):
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        q1 = Critic(hidden_dim=self.hidden_dim, num_layers=self.num_layers, name="q1")(state, action)
        q2 = Critic(hidden_dim=self.hidden_dim, num_layers=self.num_layers, name="q2")(state, action)
        return q1, q2
