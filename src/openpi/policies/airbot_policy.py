import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# TODO remove hardcoding
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
    # "halt",
    # "hold still",
]


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    """Inputs for the Airbot policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    # Whether to randomly choose prompt in TASK_AUGMENTATION, otherwise use the first one.
    prompt_augmentation: bool = False

    # Probability replace the action with state, and replace the prompt with HALT_COMMANDS.
    halt_injection_prob: float = 0.0

    # Pad action after task prompt changed according to task_len
    # |   task m    |     task n   |
    #       |action horizon|
    #               | pad  |
    pad_action: bool = False

    # Crop input image into square for siglip input are square
    crop_img_square: bool = False

    # Probability mask the wrist cam to enforce the model use front cam
    mask_wrist_cam_prob: float = 0.0

    def parse_image(self, image) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")

        if self.crop_img_square:
            h, w, _ = image.shape
            assert w > h
            left = (w - h) // 2
            right = left + h
            image = image[:, left:right, :]

        return image

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        if "images" in data:
            in_images = data["images"]

            # Assume that base image always exists.
            base_image = self.parse_image(in_images["cam_high"])

            images = {
                "base_0_rgb": base_image,
            }
            image_masks = {
                "base_0_rgb": np.True_,
            }

            # Add the extra images.
            extra_image_names = {
                "left_wrist_0_rgb": "cam_left_wrist",
                "right_wrist_0_rgb": "cam_right_wrist",
            }
            for dest, source in extra_image_names.items():
                if np.random.uniform() > self.mask_wrist_cam_prob:
                    images[dest] = self.parse_image(in_images[source])
                    image_masks[dest] = np.True_
                else:
                    images[dest] = np.zeros_like(base_image)
                    image_masks[dest] = np.False_ if mask_padding else np.True_
        else:
            # 数据集未加载图像,如计算统计信息时
            images = None
            image_masks = None

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data and data["prompt"].isupper():
            if data["prompt"] not in TASK_AUGMENTATION:
                raise ValueError(f"prompt should be keys of TASK_AUGMENTATION, got: {data['prompt']}")
            if self.prompt_augmentation:
                inputs["prompt"] = np.random.choice(TASK_AUGMENTATION[data["prompt"]])
            else:
                inputs["prompt"] = TASK_AUGMENTATION[data["prompt"]][0]
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]
            if self.pad_action and "actions" in inputs and data["task_len"] + 1 < inputs["actions"].shape[0]:
                inputs["actions"][data["task_len"] + 1:] = inputs["actions"][data["task_len"]]
        else:
            # will get prompt from InjectDefaultPrompt
            pass

        if "actions" in data and np.random.uniform() < self.halt_injection_prob:
            inputs["prompt"] = np.random.choice(HALT_COMMANDS)
            inputs["actions"][:] = state

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):
    """Outputs for the Airbot policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
