import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


TASK_AUGMENTATION = {
    "PICK_PLACE": [
        "Use right arm to pick up the block on the table and place it in the red square area.",
        "Place the block in the red square with right arm, keep left arm still.",
    ],
    "TRANSFER_BLOCK": [
        "Pick up the block with the closest hand, give it to the other hand and place it.",
    ],
    "STACK_BLOCK": [
        "Use left and right arm to stack the three blocks in the red rectangle.",
        "Stack the three blocks on top of each other in the red square with dual arms.",
    ],
    "STACK_PAPER_CUPS": [
        "Nest all paper cups together.",
    ],
    "FOLD_TOWEL": [
        "Flatten the towel and fold it along the long side.",
    ],
    "ORGANIZE_BLOCK": [
        "Use right arm to pick up the blocks, handed to left arm, and place them in the tray by color.",
        "Pick up the blocks with right arm, and place them in the tray with left arm.",
    ],
    "WIPE_WHITEBOARD": [
        "Wipe the whiteboard clean with right arm.",
    ],
}

HALT_COMMANDS = [
    "stop moving",
]


def make_airbot_example() -> dict:
    """Creates a random observation for the Airbot policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "PICK_PLACE",
    }


def _parse_image(image, *, crop_square: bool = False) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    if crop_square and image.ndim == 3:
        height, width = image.shape[:2]
        crop = min(height, width)
        top = (height - crop) // 2
        left = (width - crop) // 2
        image = image[top : top + crop, left : left + crop]

    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    """Inputs for the Airbot policy."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05
    prompt_augmentation: bool = False
    halt_injection_prob: float = 0.0
    pad_action: bool = False
    crop_img_square: bool = False
    mask_wrist_cam_prob: float = 0.0

    def _padding_mask_value(self) -> np.bool_:
        return np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)

        in_images = data.get("images", {})
        base_image = (
            _parse_image(in_images["cam_high"], crop_square=self.crop_img_square)
            if "cam_high" in in_images
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        missing_mask = self._padding_mask_value()

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_ if "cam_high" in in_images else missing_mask}
        for dest, source in {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }.items():
            use_image = source in in_images and np.random.uniform() >= self.mask_wrist_cam_prob
            if use_image:
                images[dest] = _parse_image(in_images[source], crop_square=self.crop_img_square)
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = missing_mask

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(np.asarray(data["actions"]), self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            if isinstance(prompt, str) and prompt.isupper():
                if prompt not in TASK_AUGMENTATION:
                    raise ValueError(f"prompt should be one of {tuple(TASK_AUGMENTATION)}, got {prompt}")
                prompts = TASK_AUGMENTATION[prompt]
                inputs["prompt"] = np.random.choice(prompts) if self.prompt_augmentation else prompts[0]
            else:
                inputs["prompt"] = prompt

            if self.pad_action and "actions" in inputs and "task_len" in data:
                task_len = int(np.asarray(data["task_len"]).item())
                if task_len + 1 < inputs["actions"].shape[0]:
                    inputs["actions"][task_len + 1 :] = inputs["actions"][task_len]

        if "actions" in inputs and np.random.uniform() < self.halt_injection_prob:
            inputs["prompt"] = np.random.choice(HALT_COMMANDS)
            inputs["actions"][:] = state

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):
    """Outputs for the Airbot policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
