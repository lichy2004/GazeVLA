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
import logging
import safetensors.torch
import jax
from utils import transforms as tfs
from utils.dataset_utils import HAND_JOINTS, visualize_gaze_point

import openpi.transforms as _transforms
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.models.IntentionVLA_config
import openpi.models_pytorch.IntentionVLA_pytorch

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


def compute_rotation_error_degrees(rot6d_gt, rot6d_pred):
    """Compute rotation error in degrees from 6D rotation representation."""
    # Convert rot6d to rotation matrix
    rot6d_gt = torch.from_numpy(rot6d_gt).float()
    rot6d_pred = torch.from_numpy(rot6d_pred).float()
    
    R_gt = tfs.rotation_6d_to_matrix(rot6d_gt)      # [..., 3, 3]
    R_pred = tfs.rotation_6d_to_matrix(rot6d_pred)  # [..., 3, 3]
    
    # Compute relative rotation: R_rel = R_gt^T @ R_pred
    R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    
    # Extract angle from rotation matrix: trace(R) = 1 + 2*cos(theta)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)  # Numerical stability
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg.numpy()


def compute_hand_mse(hand_gt, hand_pred):
    """Compute MSE metrics between ground truth and predicted hand actions."""
    hand_gt = np.array(hand_gt)
    hand_pred = np.array(hand_pred)
    
    # Hand structure: left_wrist(9) + 5*left_fingers(3) + right_wrist(9) + 5*right_fingers(3) = 48
    joint_indices = {
        'left_wrist': (0, 9),      # pos(3) + rot6d(6)
        'left_thumb': (9, 12),
        'left_index': (12, 15),
        'left_middle': (15, 18),
        'left_ring': (18, 21),
        'left_little': (21, 24),
        'right_wrist': (24, 33),   # pos(3) + rot6d(6)
        'right_thumb': (33, 36),
        'right_index': (36, 39),
        'right_middle': (39, 42),
        'right_ring': (42, 45),
        'right_little': (45, 48),
    }
    
    # Position indices and rotation indices
    pos_indices = list(range(0, 3)) + list(range(9, 24)) + list(range(24, 27)) + list(range(33, 48))
    
    # Position error (in meters)
    pos_gt = hand_gt[..., pos_indices]
    pos_pred = hand_pred[..., pos_indices]
    pos_diff = pos_gt - pos_pred
    
    # Per-point Euclidean distance
    pos_diff_reshaped = pos_diff.reshape(-1, pos_diff.shape[-1] // 3, 3)  # [N, num_joints, 3]
    pos_distances = np.sqrt(np.sum(pos_diff_reshaped ** 2, axis=-1))  # [N, num_joints]
    
    position_rmse_m = np.sqrt(np.mean(pos_diff ** 2))  # RMSE in meters
    position_rmse_cm = position_rmse_m * 100  # RMSE in cm
    position_mae_cm = np.mean(pos_distances) * 100  # MAE in cm
    
    # Rotation error (in degrees) - for wrists only
    left_rot_gt = hand_gt[..., 3:9]
    left_rot_pred = hand_pred[..., 3:9]
    right_rot_gt = hand_gt[..., 27:33]
    right_rot_pred = hand_pred[..., 27:33]
    
    left_rot_error = compute_rotation_error_degrees(left_rot_gt, left_rot_pred)
    right_rot_error = compute_rotation_error_degrees(right_rot_gt, right_rot_pred)
    
    rotation_mae_deg = np.mean(np.concatenate([left_rot_error.flatten(), right_rot_error.flatten()]))
    rotation_std_deg = np.std(np.concatenate([left_rot_error.flatten(), right_rot_error.flatten()]))
    
    # Per-joint position RMSE (cm)
    per_joint_rmse_cm = {}
    for joint_name, (start, end) in joint_indices.items():
        if 'wrist' in joint_name:
            joint_pos = hand_gt[..., start:start+3] - hand_pred[..., start:start+3]
        else:
            joint_pos = hand_gt[..., start:end] - hand_pred[..., start:end]
        joint_rmse = np.sqrt(np.mean(joint_pos ** 2)) * 100
        per_joint_rmse_cm[joint_name] = float(joint_rmse)
    
    return {
        'position_rmse_cm': float(position_rmse_cm),
        'position_mae_cm': float(position_mae_cm),
        'rotation_mae_deg': float(rotation_mae_deg),
        'rotation_std_deg': float(rotation_std_deg),
        'per_joint_rmse_cm': per_joint_rmse_cm,
    }


def compute_gaze_mse(gaze_gt, gaze_pred, image_size=224):
    """Compute gaze error metrics with normalized and interpretable values."""
    gaze_gt = np.array(gaze_gt)
    gaze_pred = np.array(gaze_pred)
    
    # Compute Euclidean distance for each sample
    diff = gaze_gt - gaze_pred
    
    # Distance in normalized coordinates (0-1 range)
    # Note: if gaze is already in pixel coordinates, we normalize it first
    if np.max(gaze_gt) > 1.0 or np.max(gaze_pred) > 1.0:
        # Data appears to be in pixel coordinates, normalize
        gaze_gt_norm = gaze_gt / image_size
        gaze_pred_norm = gaze_pred / image_size
        diff_norm = gaze_gt_norm - gaze_pred_norm
    else:
        # Data is already normalized
        gaze_gt_norm = gaze_gt
        gaze_pred_norm = gaze_pred
        diff_norm = diff
    
    # Euclidean distance in normalized space (0-1)
    dist_norm = np.sqrt(np.sum(diff_norm ** 2, axis=-1))
    
    # Distance in pixels (for 224x224 image)
    dist_pixels = dist_norm * image_size
    
    # Distance as percentage of image diagonal
    # Diagonal = sqrt(224^2 + 224^2) = 224 * sqrt(2) ≈ 316.8
    diagonal = image_size * np.sqrt(2)
    dist_percent_diagonal = (dist_norm * image_size / diagonal) * 100
    
    return {
        'mean_distance_pixels': float(np.mean(dist_pixels)),
        'std_distance_pixels': float(np.std(dist_pixels)),
        'mean_distance_percent': float(np.mean(dist_percent_diagonal)),  # % of diagonal
        'std_distance_percent': float(np.std(dist_percent_diagonal)),
        'median_distance_pixels': float(np.median(dist_pixels)),
    }


def save_quantitative_results(hand_metrics, gaze_metrics, save_path, config_info=None):
    """Save quantitative analysis results to markdown file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("# Evaluation Results\n\n")
        
        # Configuration section
        if config_info:
            f.write("## Configuration\n\n")
            for key, value in config_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        # Hand metrics
        f.write("## Hand Action Metrics\n\n")
        f.write("### Position Error\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| RMSE | {hand_metrics['position_rmse_cm']:.2f} cm |\n")
        f.write(f"| MAE | {hand_metrics['position_mae_cm']:.2f} cm |\n")
        f.write("\n")
        
        f.write("### Rotation Error (Wrist)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| MAE | {hand_metrics['rotation_mae_deg']:.2f}° |\n")
        f.write(f"| Std | {hand_metrics['rotation_std_deg']:.2f}° |\n")
        f.write("\n")
        
        # Per-joint RMSE
        f.write("### Per-Joint Position RMSE\n\n")
        f.write("| Joint | RMSE (cm) |\n")
        f.write("|-------|----------|\n")
        for joint_name, rmse_value in hand_metrics['per_joint_rmse_cm'].items():
            f.write(f"| {joint_name} | {rmse_value:.2f} |\n")
        f.write("\n")
        
        # Gaze metrics
        f.write("## Gaze Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Distance | {gaze_metrics['mean_distance_pixels']:.2f} px |\n")
        f.write(f"| Std Distance | {gaze_metrics['std_distance_pixels']:.2f} px |\n")
        f.write(f"| Median Distance | {gaze_metrics['median_distance_pixels']:.2f} px |\n")
        f.write(f"| Mean Distance | {gaze_metrics['mean_distance_percent']:.2f}% (of diagonal) |\n")
        f.write(f"| Std Distance | {gaze_metrics['std_distance_percent']:.2f}% (of diagonal) |\n")
    
    logging.info(f"Quantitative results saved to: {save_path}")


def visualize_on_image(image, gaze_gt=None, gaze_pred=None, hand_gt=None, hand_pred=None, 
                       render_mano=False, image_size=224):
    """Visualize gaze and hand on image."""
    image = image.copy()
    
    # Draw gaze points
    if gaze_gt is not None:
        gaze_gt_pixel = (gaze_gt * image_size).astype(int)
        cv2.circle(image, (gaze_gt_pixel[1], gaze_gt_pixel[0]), 5, (0, 255, 0), -1)  # Green for GT
    
    if gaze_pred is not None:
        gaze_pred_pixel = (gaze_pred * image_size).astype(int)
        cv2.circle(image, (gaze_pred_pixel[1], gaze_pred_pixel[0]), 5, (0, 0, 255), -1)  # Red for Pred
    
    # MANO rendering placeholder
    if render_mano and (hand_gt is not None or hand_pred is not None):
        # TODO: Implement MANO hand model rendering
        pass
    
    return image


def save_qualitative_results(save_dir, images, gaze_gt, gaze_pred, hand_gt, hand_pred,
                              num_samples=None):
    """Save qualitative analysis results."""
    os.makedirs(save_dir, exist_ok=True)
    
    bs = images.shape[0]
    num_samples = num_samples if num_samples is not None else bs
    num_samples = min(num_samples, bs)
    
    for i in range(num_samples):
        sample_dir = os.path.join(save_dir, f"batch_{i}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Visualize gaze on image
        image_with_gaze = visualize_on_image(
            images[i], 
            gaze_gt=gaze_gt[i] if gaze_gt is not None else None,
            gaze_pred=gaze_pred[i] if gaze_pred is not None else None,
        )
        cv2.imwrite(os.path.join(sample_dir, "image_gaze.png"), image_with_gaze)
        
        # Placeholder for MANO rendering
        image_with_hand = visualize_on_image(
            images[i],
            hand_gt=hand_gt[i] if hand_gt is not None else None,
            hand_pred=hand_pred[i] if hand_pred is not None else None,
            render_mano=True,
        )
        cv2.imwrite(os.path.join(sample_dir, "image_hand.png"), image_with_hand)
        
        # Save data as npz
        np.savez(
            os.path.join(sample_dir, "data.npz"),
            image=images[i],
            gaze_gt=gaze_gt[i] if gaze_gt is not None else None,
            gaze_pred=gaze_pred[i] if gaze_pred is not None else None,
            hand_gt=hand_gt[i] if hand_gt is not None else None,
            hand_pred=hand_pred[i] if hand_pred is not None else None,
        )
    
    logging.info(f"Qualitative results saved to: {save_dir}")


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

    # Compute quantitative metrics
    logging.info("Computing quantitative metrics...")
    hand_metrics = compute_hand_mse(actions_hand_gt, actions_hand_pred)
    gaze_metrics = compute_gaze_mse(gaze_gt, gaze_pred)
    
    # Save quantitative results
    quantitative_save_path = os.path.join(args.save_path, "quantitative_results.md")
    config_info = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'batch_size': batch_size,
        'num_steps': args.num_steps,
    }
    save_quantitative_results(hand_metrics, gaze_metrics, quantitative_save_path, config_info)
    
    # Save qualitative results
    if args.save_qualitative:
        logging.info("Saving qualitative results...")
        qualitative_save_dir = os.path.join(args.save_path, "qualitative")
        save_qualitative_results(
            qualitative_save_dir,
            cam_high_transformed,
            gaze_gt, gaze_pred,
            actions_hand_gt, actions_hand_pred,
            num_samples=args.num_save_samples,
        )

    # Save data to HDF5
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
    
    save_path = os.path.join(args.save_path, "eval_result.h5")
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
    
    logging.info(f"HDF5 data saved to: {save_path}")
    logging.info("=" * 80)
    logging.info("✅ Data generation and inference completed!")
    logging.info("=" * 80)
    
    return save_path


def load_from_hdf5(hdf5_path):
    """Load evaluation data from HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        fps = f["/metadata/fps"][()]
        frames = f["/metadata/frames"][()]
        batch_size = f["/metadata/batch_size"][()] if "/metadata/batch_size" in f else 1
        
        obs_images = []
        for data in f["/observations/cam_high"][:]:
            data = np.frombuffer(data, np.uint8)
            obs_images.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
        obs_images = np.array(obs_images)
        
        gaze_gt = f["/action/gaze_gt"][:]
        gaze_pred = f["/action/gaze_pred"][:]
        hand_gt = f["/action/hand_gt"][:]
        hand_pred = f["/action/hand_pred"][:]
    
    return {
        'fps': fps,
        'frames': frames,
        'batch_size': batch_size,
        'images': obs_images,
        'gaze_gt': gaze_gt,
        'gaze_pred': gaze_pred,
        'hand_gt': hand_gt,
        'hand_pred': hand_pred,
    }


def analyze_from_saved(args):
    """Load saved data and perform quantitative/qualitative analysis."""
    init_logging()
    logging.info("=" * 80)
    logging.info("Loading saved data for analysis")
    logging.info("=" * 80)
    
    data = load_from_hdf5(args.load)
    
    # Compute quantitative metrics
    logging.info("Computing quantitative metrics...")
    hand_metrics = compute_hand_mse(data['hand_gt'], data['hand_pred'])
    gaze_metrics = compute_gaze_mse(data['gaze_gt'], data['gaze_pred'])
    
    # Save quantitative results
    quantitative_save_path = os.path.join(args.save_path, "quantitative_results.md")
    config_info = {
        'load_path': args.load,
        'batch_size': data['batch_size'],
    }
    save_quantitative_results(hand_metrics, gaze_metrics, quantitative_save_path, config_info)
    
    # Save qualitative results
    if args.save_qualitative:
        logging.info("Saving qualitative results...")
        qualitative_save_dir = os.path.join(args.save_path, "qualitative")
        save_qualitative_results(
            qualitative_save_dir,
            data['images'],
            data['gaze_gt'], data['gaze_pred'],
            data['hand_gt'], data['hand_pred'],
            num_samples=args.num_save_samples,
        )
    
    logging.info("=" * 80)
    logging.info("✅ Analysis completed!")
    logging.info("=" * 80)
    
    return data


def visualize_data(args, data):
    """Visualize data using viser. data can be a dict or hdf5 path."""
    if isinstance(data, str):
        data = load_from_hdf5(data)
    
    fps = data['fps']
    frames = data['frames']
    batch_size = data['batch_size']
    obs_images = data['images']
    gaze_gt = data['gaze_gt']
    gaze_pred = data['gaze_pred']
    hand_gt = data['hand_gt']
    hand_pred = data['hand_pred']

    server = viser.ViserServer(label="Eval Visualization", host=args.server_host, port=args.server_port)
    server.gui.add_markdown(f"batch_size: {batch_size}, frames: {frames}")
    _print_server_info(args.server_host, args.server_port)
    
    while True:
        for batch_idx in range(batch_size):
            for frame in range(frames):
                image = obs_images[batch_idx].copy()
                
                # Visualize gaze
                image = visualize_gaze_point(
                    gaze_point_gt=gaze_gt[batch_idx], 
                    gaze_point_pred=gaze_pred[batch_idx], 
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

                # Visualize hand GT (green)
                _render_hand(server, hand_gt[batch_idx, frame], suffix="_gt", color=(0, 255, 0))
                
                # Visualize hand pred (red)
                _render_hand(server, hand_pred[batch_idx, frame], suffix="_pred", color=(255, 0, 0))
                
                time.sleep(1/fps)
            server.scene.reset()


def _render_hand(server, qpos, suffix="", color=(0, 255, 0)):
    """Render hand joints on viser server."""
    idx = 0
    for key in HAND_JOINTS:
        if 'wrist' in key:
            pos = qpos[idx:idx+3]
            rot = qpos[idx+3:idx+9]
            rot = tfs.rotation_6d_to_matrix(torch.from_numpy(rot))
            rot = tfs.matrix_to_quaternion(rot).numpy()
            server.scene.add_frame(
                name=f"/{key}{suffix}",
                wxyz=rot,
                position=pos,
                axes_length=0.05,
                axes_radius=0.005,
            )
            idx += 9
        else:
            pos = qpos[idx:idx+3]
            server.scene.add_icosphere(
                name=f"/{key}{suffix}",
                radius=0.01,
                color=color,
                position=pos,
            )
            idx += 3


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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_path = os.path.join(script_dir, "data")
    
    parser = argparse.ArgumentParser(description="Evaluate pretrained model on human data")
    parser.add_argument("--config", type=str, default="pretrain_hand")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hand/hand/35000")
    parser.add_argument("--save_path", type=str, default=default_save_path,
                        help="Directory to save evaluation results")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--server_host", type=str, default=SERVER_HOST)
    parser.add_argument("--server_port", type=int, default=SERVER_PORT)
    parser.add_argument("--save_qualitative", action="store_true",
                        help="Save qualitative analysis results")
    parser.add_argument("--num_save_samples", type=int, default=10,
                        help="Number of samples to save for qualitative analysis")
    parser.add_argument("--skip_viser", action="store_true", help="Skip viser visualization")
    parser.add_argument("--load", type=str, default=None,
                        help="Load saved HDF5 file for analysis (skip inference)")
    args = parser.parse_args()
    
    if args.load:
        # Load from saved data and analyze
        data = analyze_from_saved(args)
        if not args.skip_viser:
            visualize_data(args, data)
    else:
        # Run inference and save results
        save_path = generate_data(args)
        if not args.skip_viser:
            visualize_data(args, save_path)


if __name__ == "__main__":
    main()
