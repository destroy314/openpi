import dataclasses
import logging
import pathlib
import shutil
from typing import Any

from etils import epath
import jax
import orbax.checkpoint as ocp

import openpi.shared.normalize as _normalize


@dataclasses.dataclass(frozen=True)
class RLTCheckpointState:
    policy_state: dict[str, Any]
    critic_state: dict[str, Any]


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, overwrite: bool, resume: bool
) -> tuple[pathlib.Path, bool]:
    path = pathlib.Path(checkpoint_dir).resolve()
    resuming = False
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {path} already exists. Use --overwrite or --resume to continue."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path, resuming


def latest_step(checkpoint_dir: pathlib.Path | str) -> int | None:
    path = pathlib.Path(checkpoint_dir)
    steps = [int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()] if path.exists() else []
    return max(steps) if steps else None


def save_checkpoint(
    checkpoint_dir: pathlib.Path | str,
    step: int,
    *,
    policy_state: dict[str, Any],
    critic_state: dict[str, Any],
    norm_stats: dict[str, _normalize.NormStats] | None,
    asset_id: str | None,
) -> pathlib.Path:
    """Save Stage-2 RLT checkpoint.

    Only the RL model (actor optimizer state + critic) is persisted.
    The VLA backbone is frozen and always reloaded from the Stage-1 checkpoint,
    so there is no need to duplicate those weights here.
    """
    step_dir = pathlib.Path(checkpoint_dir) / str(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(step_dir / "policy_state", policy_state)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(step_dir / "critic_state", critic_state)
    if norm_stats is not None and asset_id is not None:
        _normalize.save(step_dir / "assets" / asset_id, norm_stats)
    logging.info("Saved RLT checkpoint to %s", step_dir)
    return step_dir


def save_bundle(checkpoint_dir: pathlib.Path | str, item: str, bundle: dict[str, Any]) -> pathlib.Path:
    item_dir = pathlib.Path(checkpoint_dir) / item
    item_dir.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(item_dir, bundle)
    return item_dir


def restore_bundle(checkpoint_dir: pathlib.Path | str, item: str) -> dict[str, Any]:
    item_dir = pathlib.Path(checkpoint_dir) / item
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(item_dir)
        return checkpointer.restore(item_dir, item=metadata)
