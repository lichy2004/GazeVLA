import os
import sys
sys.path.append('.')

import cv2
import h5py
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import logging
import safetensors.torch

import openpi.training.config as _config
import openpi.models.IntentionVLA_config
import openpi.models_pytorch.IntentionVLA_pytorch
from openpi.models import tokenizer as _tokenizer
from openpi.models.model import Observation
from openpi_client import image_tools


def init_logging():
    """Initialize logging system."""
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


def load_model(config: _config.TrainConfig, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    logging.info("Starting model loading...")
    
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
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    
    logging.info(f"Model config: action_dim={model_cfg.action_dim}, action_horizon={model_cfg.action_horizon}")
    
    model = openpi.models_pytorch.IntentionVLA_pytorch.IntentionVLA_pytorch(model_cfg).to(device)
    logging.info("Model instance created")
    
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"Loading weights: {model_path}")
    state_dict = safetensors.torch.load_file(model_path, device=str(device))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logging.info(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logging.info(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()
    logging.info("Model loaded and set to eval mode")
    
    return model, model_cfg


def load_hdf5_data(hdf5_path):
    """Load video data directly from HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        # Load instruction
        instruction = f["/observations/instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        
        # Load images (may be compressed JPEG)
        cam_high_data = f["/observations/cam_high"][:]
        
        # Check if images are compressed
        if cam_high_data.ndim == 1 or (cam_high_data.ndim == 2 and cam_high_data.dtype.kind in ['S', 'O', 'V']):
            # Compressed JPEG format - decode directly (data is stored as RGB)
            images = []
            for data in cam_high_data:
                if isinstance(data, bytes):
                    img_bytes = np.frombuffer(data, np.uint8)
                else:
                    img_bytes = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                images.append(img)
            images = np.array(images)
        else:
            # Uncompressed format [T, H, W, C]
            images = cam_high_data
        
        num_frames = len(images)
    
    return images, instruction, num_frames


def preprocess_frame(image, instruction, tokenizer, device):
    """Preprocess a single frame for model inference."""
    # Convert BGR to RGB for model input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to 224x224
    image_resized = image_tools.resize_with_pad(image_rgb, 224, 224)
    
    # Convert to [-1, 1] float32
    image_float = image_resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    
    # Add batch dimension and convert to [B, C, H, W]
    image_tensor = torch.from_numpy(image_float).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    
    # Tokenize prompt
    tokens, token_mask = tokenizer.tokenize(instruction)
    tokens = torch.from_numpy(tokens).unsqueeze(0).to(device)
    token_mask = torch.from_numpy(token_mask).unsqueeze(0).to(device)
    
    # Create dummy state (zeros)
    state = torch.zeros(1, 48, device=device)
    
    # Build Observation
    observation = Observation(
        images={"base_0_rgb": image_tensor},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool, device=device)},
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
        token_ar_mask=torch.zeros_like(tokens),
        token_loss_mask=torch.zeros(1, tokens.shape[1], dtype=torch.bool, device=device),
        gaze=None,
        action_mask=None,
        gaze_mask=None,
    )
    
    return observation


def draw_gaze_pred(image, gaze_pred):
    """Draw predicted gaze point on image (red color in BGR format)."""
    image = image.copy()
    h, w = int(gaze_pred[0]), int(gaze_pred[1])
    cv2.circle(image, (w, h), 5, (0, 0, 255), -1)  # BGR format: red = (0, 0, 255)
    return image


def run_video_inference(args):
    """Main video inference pipeline."""
    init_logging()
    logging.info("=" * 80)
    logging.info("Starting video inference pipeline")
    logging.info("=" * 80)
    
    # Load config and model
    config = _config.get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model, model_cfg = load_model(config, args.checkpoint, device)
    
    # Initialize tokenizer
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=config.model.max_token_len)
    
    # Load HDF5 data
    images, instruction, num_frames = load_hdf5_data(args.data_path)
    logging.info(f"Loaded {num_frames} frames from {args.data_path}")
    logging.info(f"Instruction: {instruction}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, "gaze_prediction.mp4")
    
    # Get image size
    h, w = images[0].shape[:2]
    video_size = f"{w}x{h}"
    logging.info(f"Video size: {video_size}")
    
    # Initialize ffmpeg (using bgr24 since OpenCV outputs BGR format)
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", video_size,
        "-framerate", str(args.fps),
        "-i", "-",
        "-pix_fmt", "yuv420p",
        "-vcodec", "libx264",
        "-crf", "23",
        video_path
    ], stdin=subprocess.PIPE)
    
    # Inference loop
    gaze_predictions = []
    for frame_idx in tqdm(range(num_frames), desc="Inference"):
        image = images[frame_idx]
        
        # Preprocess
        observation = preprocess_frame(image, instruction, tokenizer, device)
        
        # Model inference
        with torch.no_grad():
            _, gaze_pred = model.sample_actions(
                device=device,
                observation=observation,
                num_steps=args.num_steps
            )
        
        # Get gaze prediction in pixel coordinates
        gaze_pred_np = gaze_pred.detach().cpu().numpy()[0]  # [2]
        gaze_predictions.append(gaze_pred_np)
        
        # Draw gaze on image and write to video
        image_with_gaze = draw_gaze_pred(image, gaze_pred_np)
        ffmpeg.stdin.write(image_with_gaze.tobytes())
    
    # Close ffmpeg
    ffmpeg.stdin.close()
    ffmpeg.wait()
    
    # Save gaze predictions
    gaze_save_path = os.path.join(args.output_dir, "gaze_predictions.npy")
    np.save(gaze_save_path, np.array(gaze_predictions))
    
    logging.info("=" * 80)
    logging.info(f"Video saved to: {video_path}")
    logging.info(f"Gaze predictions saved to: {gaze_save_path}")
    logging.info("=" * 80)
    
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Video-level gaze inference")
    parser.add_argument(
        "--config",
        type=str,
        default="human_to_robot",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/human_to_robot/human_to_robot/30000",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/lichy/root/Datasets/IntentionVLA/hdf5/20251229/human_lemon/1.hdf5",
        help="Path to the HDF5 data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./debug/human_to_robot",
        help="Directory to save output video",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frames per second",
    )
    args = parser.parse_args()
    
    run_video_inference(args)


if __name__ == "__main__":
    main()
