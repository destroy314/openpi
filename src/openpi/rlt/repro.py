from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import tempfile
from typing import Any

import numpy as np

from openpi.rlt import checkpointing as _checkpointing
import openpi.shared.normalize as _normalize


_TRANSITION_FIELDS = (
    "state",
    "action",
    "reference_action",
    "reward",
    "next_state",
    "next_reference_action",
    "bootstrap_steps",
    "done",
    "env_step",
)


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_jsonable(value: Any) -> Any:
    return _to_jsonable(value)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_runtime_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("jax", "numpy", "flax", "optax", "orbax-checkpoint", "tyro"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    git_head = _git_output("rev-parse", "HEAD")
    git_status = _git_output("status", "--short")
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": {
            "head": git_head,
            "is_dirty": bool(git_status),
            "status": git_status.splitlines() if git_status else [],
        },
        "versions": versions,
    }


def write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> pathlib.Path:
    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = pathlib.Path(handle.name)
    os.replace(temp_path, output_path)
    return output_path


def write_manifest(repro_dir: pathlib.Path | str, payload: dict[str, Any]) -> pathlib.Path:
    return write_json(pathlib.Path(repro_dir) / "manifest.json", payload)


def load_manifest(repro_dir: pathlib.Path | str) -> dict[str, Any]:
    return json.loads((pathlib.Path(repro_dir) / "manifest.json").read_text(encoding="utf-8"))


def save_resolved_env_config(repro_dir: pathlib.Path | str, env_config: Any) -> pathlib.Path:
    payload = dataclasses.asdict(env_config) if dataclasses.is_dataclass(env_config) else env_config
    return write_json(pathlib.Path(repro_dir) / "resolved_env_config.json", payload)


def save_norm_stats_snapshot(
    repro_dir: pathlib.Path | str,
    norm_stats: dict[str, _normalize.NormStats] | None,
    asset_id: str | None,
) -> pathlib.Path | None:
    if norm_stats is None or asset_id is None:
        return None
    asset_dir = pathlib.Path(repro_dir) / "assets" / asset_id
    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    _normalize.save(asset_dir, norm_stats)
    return asset_dir


def save_initial_state(repro_dir: pathlib.Path | str, item: str, bundle: dict[str, Any]) -> pathlib.Path:
    return _checkpointing.save_bundle(pathlib.Path(repro_dir) / "initial_state", item, bundle)


def load_initial_state(repro_dir: pathlib.Path | str, item: str) -> dict[str, Any]:
    return _checkpointing.restore_bundle(pathlib.Path(repro_dir) / "initial_state", item)


def save_stop_signal(repro_dir: pathlib.Path | str, final_step: int) -> pathlib.Path:
    return write_json(pathlib.Path(repro_dir) / "stop_signal.json", {"final_step": int(final_step)})


def load_stop_signal(repro_dir: pathlib.Path | str) -> dict[str, Any]:
    return json.loads((pathlib.Path(repro_dir) / "stop_signal.json").read_text(encoding="utf-8"))


class TransitionRecorder:
    def __init__(self, repro_dir: pathlib.Path | str, *, shard_size: int = 1024) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self._repro_dir = pathlib.Path(repro_dir)
        self._transitions_dir = self._repro_dir / "transitions"
        self._transitions_dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = int(shard_size)
        self._buffer: list[dict[str, Any]] = []
        self._shard_index = 0

    def add(self, item: Any) -> None:
        self._buffer.append({field: getattr(item, field) for field in _TRANSITION_FIELDS})
        if len(self._buffer) >= self._shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        shard_path = self._transitions_dir / f"transitions_{self._shard_index:06d}.npz"
        payload = {
            "state": np.stack([np.asarray(entry["state"], dtype=np.float32) for entry in self._buffer], axis=0),
            "action": np.stack([np.asarray(entry["action"], dtype=np.float32) for entry in self._buffer], axis=0),
            "reference_action": np.stack(
                [np.asarray(entry["reference_action"], dtype=np.float32) for entry in self._buffer], axis=0
            ),
            "reward": np.asarray([entry["reward"] for entry in self._buffer], dtype=np.float32),
            "next_state": np.stack([np.asarray(entry["next_state"], dtype=np.float32) for entry in self._buffer], axis=0),
            "next_reference_action": np.stack(
                [np.asarray(entry["next_reference_action"], dtype=np.float32) for entry in self._buffer], axis=0
            ),
            "bootstrap_steps": np.asarray([entry["bootstrap_steps"] for entry in self._buffer], dtype=np.int32),
            "done": np.asarray([entry["done"] for entry in self._buffer], dtype=bool),
            "env_step": np.asarray([entry["env_step"] for entry in self._buffer], dtype=np.int64),
        }
        with tempfile.NamedTemporaryFile("wb", dir=self._transitions_dir, delete=False) as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = pathlib.Path(handle.name)
        os.replace(temp_path, shard_path)
        self._buffer.clear()
        self._shard_index += 1

    def close(self) -> None:
        self.flush()


def iter_transition_dicts(repro_dir: pathlib.Path | str):
    transitions_dir = pathlib.Path(repro_dir) / "transitions"
    for shard_path in sorted(transitions_dir.glob("transitions_*.npz")):
        with np.load(shard_path) as shard:
            count = int(shard["reward"].shape[0])
            for index in range(count):
                yield {
                    "state": np.asarray(shard["state"][index], dtype=np.float32),
                    "action": np.asarray(shard["action"][index], dtype=np.float32),
                    "reference_action": np.asarray(shard["reference_action"][index], dtype=np.float32),
                    "reward": float(shard["reward"][index]),
                    "next_state": np.asarray(shard["next_state"][index], dtype=np.float32),
                    "next_reference_action": np.asarray(shard["next_reference_action"][index], dtype=np.float32),
                    "bootstrap_steps": int(shard["bootstrap_steps"][index]),
                    "done": bool(shard["done"][index]),
                    "env_step": int(shard["env_step"][index]),
                }


class ExecutionRecordRecorder:
    def __init__(self, repro_dir: pathlib.Path | str) -> None:
        self._path = pathlib.Path(repro_dir) / "execution_records.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def add_chunk(
        self,
        *,
        episode_id: str,
        env_step: int,
        action_sources: list[str],
        records: list[dict[str, Any]],
        feature_metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "episode_id": episode_id,
            "env_step": int(env_step),
            "action_sources": list(action_sources),
            "records": records,
        }
        if feature_metadata is not None:
            payload["feature_metadata"] = feature_metadata
        self._handle.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def iter_execution_record_chunks(repro_dir: pathlib.Path | str):
    path = pathlib.Path(repro_dir) / "execution_records.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)
