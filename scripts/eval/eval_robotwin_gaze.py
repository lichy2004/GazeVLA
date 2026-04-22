import os
import sys
sys.path.append('.')

import cv2
import time
import h5py
import torch
import argparse
import numpy as np
import viser
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import safetensors.torch
import jax
from utils import transforms as tfs
from utils.dataset_utils import *

import openpi.models.gemma as _gemma
import openpi.transforms as _transforms
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.models.IntentionVLA_config
import openpi.models_pytorch.IntentionVLA_pytorch
import openpi.models_pytorch.preprocessing_human as _preprocessing

matplotlib.use('Agg')
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080


def init_logging():
    """Initialize logging system, maintaining consistent format with training script"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def convert_delta_to_absolute_actions(actions, state, mask):
    mask = np.asarray(mask)
    dims = mask.shape[-1]
    actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    return actions


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def denormalize_data(data, norm_stats=None, key="actions"):
    if norm_stats is None or key not in norm_stats:
        print(f"Warning: No normalization statistics found for {key}, returning original data")
        return data
    
    # Convert to numpy array for processing
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.array(data)
    
    stats = norm_stats[key]
    mean = stats.mean
    std = stats.std
    
    # Ensure dimension matching
    if mean.shape[-1] < data_np.shape[-1]:
        # If statistics dimension is smaller than data dimension, pad mean with 0 and std with 1
        mean_padded = np.pad(mean, (0, data_np.shape[-1] - mean.shape[-1]), 'constant', constant_values=0.0)
        std_padded = np.pad(std, (0, data_np.shape[-1] - std.shape[-1]), 'constant', constant_values=1.0)
    else:
        mean_padded = mean
        std_padded = std
    
    # Denormalize: x = (x_normalized * std) + mean
    denormalized = data_np * (std_padded + 1e-6) + mean_padded
    
    return denormalized


def transform_images(images):
    # Convert to numpy array
    if isinstance(images, torch.Tensor):
        images_np = images.detach().cpu().numpy()
    else:
        images_np = np.array(images)
    
    # Convert from [-1, 1] to [0, 255]
    images_np = ((images_np + 1.0) * 127.5).astype(np.uint8)
    
    # Transform dimensions: [batch_size, channels, height, width] -> [batch_size, height, width, channels]
    if len(images_np.shape) == 4:
        images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    return images_np


def _preprocess_observation(observation, *, train=True):
    """Helper method to preprocess observation."""
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
    return (
        list(observation.images.values()),
        list(observation.image_masks.values()),
        observation.tokenized_prompt,
        observation.tokenized_prompt_mask,
        observation.state,
        observation.gaze,
        observation.gaze_masks,
    )


def load_model(config: _config.TrainConfig, checkpoint_path: str, device: torch.device):
    logging.info("Starting model loading...")
    
    # 1. Create model configuration
    if not isinstance(config.model, openpi.models.IntentionVLA_config.IntentionVLAConfig):
        model_cfg = openpi.models.IntentionVLA_config.IntentionVLAConfig(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    
    logging.info(f"Model configuration: action_dim={model_cfg.action_dim}, action_horizon={model_cfg.action_horizon}")
    
    # 2. Create model instance
    model = openpi.models_pytorch.IntentionVLA_pytorch.IntentionVLA_pytorch(model_cfg).to(device)
    logging.info("Model instance creation completed")
    
    # 3. Load checkpoint weights
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    logging.info(f"Loading weights: {model_path}")
    
    # 使用 load_file 加载权重字典，然后用 strict=False 加载到模型
    # 这样可以忽略新增的 gaze CoT 相关层（如果 checkpoint 中不存在）
    state_dict = safetensors.torch.load_file(model_path, device=str(device))
    projection_bases = ["action_in_proj", "action_out_proj", "state_proj", "gaze_head"]
    skipped_keys = []
    for base in projection_bases:
        for suffix in ("weight", "bias"):
            key = f"{base}.{suffix}"
            if key in state_dict:
                if state_dict[key].shape != model.state_dict()[key].shape:
                    skipped_keys.append(key)
                    del state_dict[key]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logging.info(f"Missing keys (will be randomly initialized): {missing_keys}")
    if unexpected_keys:
        logging.info(f"Unexpected keys (will be ignored): {unexpected_keys}")
    
    logging.info("Model weights loading completed")
    
    # 4. Set to evaluation mode
    model.eval()
    logging.info("Model has been set to evaluation mode")
    
    return model, model_cfg


def generate_data(args):
    # Initialize logging
    init_logging()
    logging.info("=" * 80)
    logging.info("Starting data generation and inference pipeline")
    logging.info("=" * 80)
    
    # Initialize configuration and data loader
    config = _config.get_config(args.config)
    object.__setattr__(config, 'batch_size', args.batch_size)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    data_config = loader.data_config()
    norm_stats = data_config.norm_stats
    use_delta_joint_actions = config.data.use_delta_joint_actions

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint}")
    
    # Load model
    model, model_cfg = load_model(config, args.checkpoint, device)
    
    # Get data
    logging.info("Sampling data from dataset...")
    observation, actions = next(iter(loader))
    batch_size = observation.state.shape[0]
    logging.info(f"Data retrieval successful, batch_size={batch_size}")
    
    # Transfer data to device
    logging.info(f"Transferring data to device: {device}")
    observation = jax.tree.map(lambda x: x.to(device), observation)
    
    # Perform inference using model
    logging.info(f"Starting inference, diffusion steps={args.num_steps}...")
    with torch.no_grad():
        action_pred, gaze_pred = model.sample_actions(
            device=device, 
            observation=observation, 
            num_steps=args.num_steps
        )
    logging.info(f"Inference completed")
    

    # Gaze
    gaze_pred = gaze_pred.detach().cpu().numpy()    # [bs, 2]
    cam_high = observation.images['base_0_rgb']     # [bs, c, h, w]
    cam_high_transformed = transform_images(cam_high)
    bs = gaze_pred.shape[0]
    
    save_dir = "./debug/robotwin_gaze"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(bs):
        gaze = gaze_pred[i]
        cam_high = cam_high_transformed[i]
        image = visualize_gaze_point(
            gaze_point_gt=gaze, 
            gaze_point_pred=gaze, 
            image=cam_high, 
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path = os.path.join(save_dir, f"{str(i).zfill(6)}.jpg")
        cv2.imwrite(path, image)
    print(f"Saved {bs} images to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune_robotwin_single",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="./debug/eval_gaze/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gaze/gaze/20000",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--server_host",
        type=str,
        default=SERVER_HOST,
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=SERVER_PORT,
    )
    args = parser.parse_args()
    
    save_path = "./debug/eval_gaze/eval_result.h5"
    save_path = generate_data(args)


if __name__ == "__main__":
    main()
