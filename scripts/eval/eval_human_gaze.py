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
    
    state_dict = safetensors.torch.load_file(model_path, device=str(device))
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
    gaze_gt = observation.gaze.detach().cpu().numpy()   # [bs, 2]
    gaze_pred = gaze_pred.detach().cpu().numpy()        # [bs, 2]

    # Hand
    states_hand = observation.state # [bs, 48]
    states_hand = denormalize_data(states_hand, norm_stats, key="state")

    actions_hand_gt = actions          # [bs, 30, 48]
    actions_hand_gt = denormalize_data(actions_hand_gt, norm_stats, key="actions")
    if use_delta_joint_actions:
        delta_mask = _transforms.make_bool_mask(48)
        actions_hand_gt = convert_delta_to_absolute_actions(actions_hand_gt, states_hand, delta_mask)
    
    actions_hand_pred = action_pred
    actions_hand_pred = denormalize_data(actions_hand_pred, norm_stats, key="actions")
    if use_delta_joint_actions:
        actions_hand_pred = convert_delta_to_absolute_actions(actions_hand_pred, states_hand, delta_mask)
    
    # Image
    cam_high = observation.images['base_0_rgb']  # [bs, c, h, w]
    cam_high_transformed = transform_images(cam_high)

    # Save data
    bs, frames, dim = actions_hand_gt.shape
    data = {
        'dataset_name': 'debug',
        'data_id': 'debug',
        'robot': 'human',
        'fps': 10,
        'frames': frames,
        'if_hand': True,
        'if_gaze': True,
    }
    
    # Construct save path
    save_path = os.path.join(args.hdf5_path, "eval_result.h5")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        metadata = f.create_group("metadata")
        metadata.create_dataset("dataset_name", data=data['dataset_name'])
        metadata.create_dataset("data_id", data=data['data_id'])
        metadata.create_dataset("robot", data=data['robot'])
        metadata.create_dataset("fps", data=data['fps'])
        metadata.create_dataset("frames", data=data['frames'])
        metadata.create_dataset("batch_size", data=config.batch_size)
        metadata.create_dataset("if_hand", data=data['if_hand'])
        metadata.create_dataset("if_gaze", data=data['if_gaze'])
        
        action = f.create_group("action")
        action.create_dataset("hand_gt", data=actions_hand_gt)      # [bs, frames, dim]
        action.create_dataset("hand_pred", data=actions_hand_pred)  # [bs, frames, dim]
        action.create_dataset("gaze_gt", data=gaze_gt)      # [bs, 2]
        action.create_dataset("gaze_pred", data=gaze_pred)  # [bs, 2]
        
        obs = f.create_group("observations")
        obs.create_dataset("state", data=states_hand)        # [bs, dim]
        obs.create_dataset("instruction", data="")
        
        cam_high_enc_list = []
        max_len = 0
        for i in range(bs):
            enc_data, enc_len = images_encoding([cam_high_transformed[i]])
            cam_high_enc_list.extend(enc_data)
            max_len = max(max_len, enc_len)
        cam_high_enc_padded = [data.ljust(max_len, b"\0") for data in cam_high_enc_list]
        obs.create_dataset("cam_high", data=cam_high_enc_padded, dtype=f"S{max_len}")
    
    logging.info(f"Data successfully saved to: {save_path}")
    logging.info("=" * 80)
    logging.info("✅ Data generation and inference completed!")
    logging.info("=" * 80)
    
    return save_path


def visualize_data(args, data_path):
    with h5py.File(data_path, "r") as f:
        dataset_name = f["/metadata/dataset_name"][()].decode('utf-8') if isinstance(f["/metadata/dataset_name"][()], bytes) else f["/metadata/dataset_name"][()]
        data_id = f["/metadata/data_id"][()].decode('utf-8') if isinstance(f["/metadata/data_id"][()], bytes) else f["/metadata/data_id"][()]
        robot = f["/metadata/robot"][()].decode('utf-8') if isinstance(f["/metadata/robot"][()], bytes) else f["/metadata/robot"][()]
        fps = f["/metadata/fps"][()]
        frames = f["/metadata/frames"][()]
        batch_size = f["/metadata/batch_size"][()] if "/metadata/batch_size" in f else 1
        if_hand = f["/metadata/if_hand"][()]
        if_gaze = f["/metadata/if_gaze"][()]

        obs_instruction = f["/observations/instruction"][()].decode('utf-8') if isinstance(f["/observations/instruction"][()], bytes) else f["/observations/instruction"][()]
        obs_state = f["/observations/state"][:]     # [bs, 48]
        obs_images = []
        for data in f["/observations/cam_high"][:]:
            data = np.frombuffer(data, np.uint8)
            obs_images.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
        obs_images = np.array(obs_images)  # [bs, H, W, C]

        gaze_gt = f["/action/gaze_gt"][:]       # [bs, 2]
        gaze_pred = f["/action/gaze_pred"][:]   # [bs, 2]
        hand_gt = f["/action/hand_gt"][:]       # [bs, frames, 48]
        hand_pred = f["/action/hand_pred"][:]   # [bs, frames, 48]

    server = viser.ViserServer(label="Remote Viser Server", host=args.server_host, port=args.server_port)
    server.gui.add_markdown(
        content=f"""
            dataset_name: {dataset_name}
            data_id: {data_id}
            batch_size: {batch_size}
            if_hand: {if_hand}, if_gaze: {if_gaze}
            Instruction: {obs_instruction}
        """,
    )
    _print_server_info(args.server_host, args.server_port)
    
    # TODO: change me
    if_hand = False 
    if_gaze = True
    
    while True:
        for batch_idx in range(batch_size):
            for frame in range(frames):
                image = obs_images[batch_idx]
                h, w, c = image.shape
                
                if if_gaze:
                    gaze_pixel_gt = gaze_gt[batch_idx]
                    gaze_pixel_pred = gaze_pred[batch_idx]
                    image = visualize_gaze_point(
                        gaze_point_gt=gaze_pixel_gt, 
                        gaze_point_pred=gaze_pixel_pred, 
                        image=image, 
                    )
                
                server.scene.add_image(
                    name="/image",
                    image=image,
                    render_width=1,
                    render_height=1,
                    format="jpeg",
                    wxyz=(0.5, -0.5, 0.5, -0.5),
                    position=(1.0, 0.0, 0.0),
                )

                # action gt, green
                if if_hand and hand_gt is not None:
                    qpos = hand_gt[batch_idx, frame]
                    idx = 0
                    for key in HAND_JOINTS:
                        if 'wrist' in key:
                            pos = qpos[idx:idx+3]
                            rot = qpos[idx+3:idx+9]
                            rot = tfs.rotation_6d_to_matrix(torch.from_numpy(rot))
                            rot = tfs.matrix_to_quaternion(rot).numpy()
                            server.scene.add_frame(
                                name=f"/{key}_hand_gt",
                                wxyz=rot,
                                position=pos,
                                axes_length=0.05,
                                axes_radius=0.005,
                            )
                            idx += 9
                        else:
                            pos = qpos[idx:idx+3]
                            server.scene.add_icosphere(
                                name=f"/{key}_hand_gt",
                                radius=0.01,
                                color=(0, 255, 0),
                                position=pos,
                            )
                            idx += 3
                
                # action pred, red
                if if_hand and hand_pred is not None:
                    qpos = hand_pred[batch_idx, frame]
                    idx = 0
                    for key in HAND_JOINTS:
                        if 'wrist' in key:
                            pos = qpos[idx:idx+3]
                            rot = qpos[idx+3:idx+9]
                            rot = tfs.rotation_6d_to_matrix(torch.from_numpy(rot))
                            rot = tfs.matrix_to_quaternion(rot).numpy()
                            server.scene.add_frame(
                                name=f"/{key}_hand_pred",
                                wxyz=rot,
                                position=pos,
                                axes_length=0.05,
                                axes_radius=0.005,
                            )
                            idx += 9
                        else:
                            pos = qpos[idx:idx+3]
                            server.scene.add_icosphere(
                                name=f"/{key}_hand_pred",
                                radius=0.01,
                                color=(255, 0, 0),
                                position=pos,
                            )
                            idx += 3
                
                time.sleep(1/fps)
            server.scene.reset()


def _print_server_info(server_host, server_port):
    # ANSI color codes
    BLUE = '\033[94m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print("=" * 60)
    print("🚀 Viser server started!")
    print(f"📍 Server address: {server_host}:{server_port}")
    print(f"      Then visit: http://localhost:{server_port}")
    print("⏹️  Press Ctrl+C to stop server")
    print("-" * 60)
    print("📊 可视化说明:")
    print(f"   {BLUE}{BOLD}绿色{RESET} = Action GT (Ground Truth)")
    print(f"   {RED}{BOLD}红色{RESET} = Action Pred (Prediction)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default="lfa-pi0-cam1-chunk25-av_aloha_sim_cube_transfer",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/lfa-pi0-cam1-chunk25-av_aloha_sim_cube_transfer/lfa-pi0-cam1-chunk25-av_aloha_sim_cube_transfer/10000",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default="./debug/eval_gaze/",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
    visualize_data(args, save_path)


if __name__ == "__main__":
    main()
