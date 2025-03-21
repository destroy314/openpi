"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
from transformers import AutoProcessor
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config.repack_transforms.inputs[0].structure.pop("images")
    dataset = _data_loader.create_dataset(data_config, config.model, disable_video=True)
    if data_config.norm_stats is None:
        raise ValueError(
            "Normalization stats not found. "
            "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
        )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        ],
    )
    num_frames = len(dataset)
    bs = 128
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=bs,
        num_workers=16,
        num_batches=num_frames // bs + 1,
    )
    actions = []
    for batch in tqdm.tqdm(data_loader, total=num_frames // bs + 1, desc="Loading actions"):
        actions.append(batch["actions"])
    actions = np.concatenate(actions, axis=0)

    outside_range = np.sum(np.abs(actions) > 1) / actions.size
    print(f"outside_range: {outside_range}")  # 0.0199 for stack_block

    print(f"fitting tokenizer for actions {actions.shape}")
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    tokenizer.fit(actions)

    if isinstance(data_config.repo_id, str):
        dataset_names = [data_config.repo_id]
    elif isinstance(data_config.repo_id, list):
        dataset_names = data_config.repo_id
    dataset_names = dataset_names[0].split("/")[0] + "/" + "_".join([name.split("/")[1] for name in dataset_names])
    print(f"saving tokenizer to {config.assets_dirs / dataset_names}")
    tokenizer.save_pretrained(config.assets_dirs / dataset_names)


if __name__ == "__main__":
    tyro.cli(main)
