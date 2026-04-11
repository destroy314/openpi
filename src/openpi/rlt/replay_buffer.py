import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class TransitionBatch:
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    next_reference_action: np.ndarray
    bootstrap_steps: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    """Simple ring buffer for RLT transitions."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self._capacity = capacity
        self._state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reference_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity,), dtype=np.float32)
        self._next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_reference_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._bootstrap_steps = np.zeros((capacity,), dtype=np.int32)
        self._done = np.zeros((capacity,), dtype=np.float32)
        self._size = 0
        self._index = 0

    def __len__(self) -> int:
        return self._size

    @property
    def state_dim(self) -> int:
        return self._state.shape[-1]

    @property
    def action_dim(self) -> int:
        return self._action.shape[-1]

    def add(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        reference_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_reference_action: np.ndarray,
        bootstrap_steps: int,
        done: bool,
    ) -> None:
        self._state[self._index] = np.asarray(state, dtype=np.float32)
        self._action[self._index] = np.asarray(action, dtype=np.float32)
        self._reference_action[self._index] = np.asarray(reference_action, dtype=np.float32)
        self._reward[self._index] = float(reward)
        self._next_state[self._index] = np.asarray(next_state, dtype=np.float32)
        self._next_reference_action[self._index] = np.asarray(next_reference_action, dtype=np.float32)
        self._bootstrap_steps[self._index] = int(bootstrap_steps)
        self._done[self._index] = float(done)

        self._index = (self._index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, *, rng: np.random.Generator | None = None) -> TransitionBatch:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples in replay buffer: need {batch_size}, have {self._size}")
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            state=self._state[indices],
            action=self._action[indices],
            reference_action=self._reference_action[indices],
            reward=self._reward[indices],
            next_state=self._next_state[indices],
            next_reference_action=self._next_reference_action[indices],
            bootstrap_steps=self._bootstrap_steps[indices],
            done=self._done[indices],
        )
