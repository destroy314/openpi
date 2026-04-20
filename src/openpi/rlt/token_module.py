import flax.linen as nn
import jax
import jax.numpy as jnp


def masked_reconstruction_loss(
    reconstruction: jax.Array, target: jax.Array, mask: jax.Array
) -> jax.Array:
    """Mean squared reconstruction loss over valid prefix tokens."""
    token_mask = mask.astype(target.dtype)[..., None]
    squared_error = jnp.square(reconstruction - target) * token_mask
    denom = jnp.clip(jnp.sum(token_mask, axis=(1, 2)) * reconstruction.shape[-1], 1.0)
    return jnp.sum(squared_error, axis=(1, 2)) / denom


def _sinusoidal_positions(length: int, dim: int, dtype: jnp.dtype) -> jax.Array:
    if dim == 0:
        return jnp.zeros((length, 0), dtype=dtype)
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    half_dim = max(dim // 2, 1)
    freq = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim)
    angles = position * freq[None, :]
    embeddings = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if embeddings.shape[-1] < dim:
        embeddings = jnp.pad(embeddings, ((0, 0), (0, dim - embeddings.shape[-1])))
    return embeddings[:, :dim].astype(dtype)


def _make_attn_mask(query_mask: jax.Array, key_mask: jax.Array) -> jax.Array:
    return jnp.logical_and(query_mask[:, :, None], key_mask[:, None, :])[:, None, :, :]


def _make_causal_attn_mask(query_mask: jax.Array, key_mask: jax.Array) -> jax.Array:
    query_len = query_mask.shape[1]
    key_len = key_mask.shape[1]
    causal = jnp.tril(jnp.ones((query_len, key_len), dtype=jnp.bool_))
    valid = jnp.logical_and(query_mask[:, :, None], key_mask[:, None, :])
    return jnp.logical_and(valid[:, None, :, :], causal[None, None, :, :])


class FeedForwardBlock(nn.Module):
    hidden_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = nn.Dense(self.mlp_dim)(inputs)
        x = nn.gelu(x)
        return nn.Dense(self.hidden_dim)(x)


class EncoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs: jax.Array, mask: jax.Array) -> jax.Array:
        x = inputs + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            deterministic=True,
        )(nn.LayerNorm()(inputs), nn.LayerNorm()(inputs), mask=mask)
        x = x + FeedForwardBlock(self.hidden_dim, self.mlp_dim)(nn.LayerNorm()(x))
        return x


class DecoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        memory: jax.Array,
        self_mask: jax.Array,
        cross_mask: jax.Array,
    ) -> jax.Array:
        x = inputs + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            deterministic=True,
        )(nn.LayerNorm()(inputs), nn.LayerNorm()(inputs), mask=self_mask)
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            deterministic=True,
        )(nn.LayerNorm()(x), nn.LayerNorm()(memory), mask=cross_mask)
        x = x + FeedForwardBlock(self.hidden_dim, self.mlp_dim)(nn.LayerNorm()(x))
        return x


class RLTokenModule(nn.Module):
    """Transformer encoder-decoder RL token bottleneck with a learned readout token."""

    input_dim: int
    token_dim: int
    depth: int = 2
    hidden_dim: int | None = None
    num_heads: int = 4
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, prefix_embeddings: jax.Array, prefix_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden_dim = self.hidden_dim or self.token_dim
        seq_len = prefix_embeddings.shape[1]
        stopped_prefix = jax.lax.stop_gradient(prefix_embeddings)

        prefix_inputs = nn.Dense(hidden_dim, name="encoder_input_proj")(prefix_embeddings)
        encoder_pos = _sinusoidal_positions(seq_len + 1, hidden_dim, prefix_inputs.dtype)
        readout_token = self.param(
            "readout_token",
            lambda key, shape: jax.random.normal(key, shape, dtype=prefix_inputs.dtype) * 0.02,
            (1, 1, hidden_dim),
        )
        readout_token = jnp.broadcast_to(readout_token, (prefix_inputs.shape[0], 1, hidden_dim))
        encoder_tokens = jnp.concatenate([prefix_inputs, readout_token], axis=1) + encoder_pos[None, :, :]
        encoder_mask = jnp.concatenate(
            [prefix_mask, jnp.ones((prefix_mask.shape[0], 1), dtype=jnp.bool_)],
            axis=1,
        )
        encoder_attn_mask = _make_attn_mask(encoder_mask, encoder_mask)
        for layer in range(self.depth):
            encoder_tokens = EncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_ratio * hidden_dim,
                name=f"encoder_block_{layer}",
            )(encoder_tokens, encoder_attn_mask)

        rl_token = nn.Dense(self.token_dim, name="token_proj")(nn.LayerNorm(name="token_ln")(encoder_tokens[:, -1]))

        memory = nn.Dense(hidden_dim, name="decoder_memory_proj")(rl_token)[:, None, :]
        bos = nn.Dense(self.input_dim, name="decoder_bos_proj")(rl_token)[:, None, :]
        shifted_targets = jnp.concatenate([bos, stopped_prefix[:, :-1]], axis=1)
        decoder_tokens = nn.Dense(hidden_dim, name="decoder_input_proj")(shifted_targets)
        decoder_tokens = decoder_tokens + _sinusoidal_positions(seq_len, hidden_dim, decoder_tokens.dtype)[None, :, :]
        decoder_self_mask = _make_causal_attn_mask(prefix_mask, prefix_mask)
        decoder_cross_mask = _make_attn_mask(prefix_mask, jnp.ones((prefix_mask.shape[0], 1), dtype=jnp.bool_))
        for layer in range(self.depth):
            decoder_tokens = DecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_ratio * hidden_dim,
                name=f"decoder_block_{layer}",
            )(decoder_tokens, memory, decoder_self_mask, decoder_cross_mask)

        reconstruction = nn.Dense(self.input_dim, name="reconstruction_head")(
            nn.LayerNorm(name="decoder_output_ln")(decoder_tokens)
        )
        return rl_token, reconstruction
