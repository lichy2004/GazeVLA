from collections.abc import Sequence
import logging
import torch
from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
#     "left_wrist_0_rgb",
#     "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

# Augmentation parameters (global variables for easy tuning)
AUG_CROP_SCALE = 0.95       # 0.95
AUG_ROTATION_ANGLE = 5      # 5
AUG_BRIGHTNESS = 0.3        # 0.3
AUG_CONTRAST = 0.4          # 0.4
AUG_SATURATION = 0.5        # 0.5


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    
    original_gaze = observation.gaze if hasattr(observation, 'gaze') and observation.gaze is not None else None
    processed_gaze = original_gaze.clone().float() if original_gaze is not None else None
    gaze_valid_mask = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device) if original_gaze is not None else None

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * AUG_CROP_SCALE)
                crop_width = int(width * AUG_CROP_SCALE)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]
                    
                    if processed_gaze is not None:
                        processed_gaze[:, 0] = processed_gaze[:, 0] - start_h
                        processed_gaze[:, 1] = processed_gaze[:, 1] - start_w

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                if processed_gaze is not None:
                    scale_h = height / crop_height
                    scale_w = width / crop_width
                    processed_gaze[:, 0] = processed_gaze[:, 0] * scale_h
                    processed_gaze[:, 1] = processed_gaze[:, 1] * scale_w

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * (2 * AUG_ROTATION_ANGLE) - AUG_ROTATION_ANGLE  # Random angle between -AUG_ROTATION_ANGLE and +AUG_ROTATION_ANGLE degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
                    
                    if processed_gaze is not None:
                        center_h = height / 2.0
                        center_w = width / 2.0
                        gaze_h_centered = processed_gaze[:, 0] - center_h
                        gaze_w_centered = processed_gaze[:, 1] - center_w
                        gaze_h_rotated = gaze_h_centered * cos_a - gaze_w_centered * sin_a
                        gaze_w_rotated = gaze_h_centered * sin_a + gaze_w_centered * cos_a
                        processed_gaze[:, 0] = gaze_h_rotated + center_h
                        processed_gaze[:, 1] = gaze_w_rotated + center_w
                        
            brightness_factor = (1.0 - AUG_BRIGHTNESS) + torch.rand(1, device=image.device) * (2 * AUG_BRIGHTNESS)  # Factor in [1-AUG_BRIGHTNESS, 1+AUG_BRIGHTNESS]
            image = image * brightness_factor

            contrast_factor = (1.0 - AUG_CONTRAST) + torch.rand(1, device=image.device) * (2 * AUG_CONTRAST)  # Factor in [1-AUG_CONTRAST, 1+AUG_CONTRAST]
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            saturation_factor = (1.0 - AUG_SATURATION) + torch.rand(1, device=image.device) * (2 * AUG_SATURATION)  # Factor in [1-AUG_SATURATION, 1+AUG_SATURATION]
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    if processed_gaze is not None:
        height, width = image_resolution
        processed_gaze = processed_gaze.float()
        processed_gaze[:, 0] = processed_gaze[:, 0] / height
        processed_gaze[:, 1] = processed_gaze[:, 1] / width
        processed_gaze = torch.clamp(processed_gaze, 0.0, 1.0)

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        action_mask=observation.action_mask,
        gaze=processed_gaze,
        gaze_mask=observation.gaze_mask,
    )
