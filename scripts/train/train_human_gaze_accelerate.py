import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

# Accelerate imports for distributed training
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from accelerate import DeepSpeedPlugin
from transformers import set_seed as transformers_set_seed
from accelerate.utils import set_seed as accelerate_set_seed

import openpi.models.IntentionVLA_config
import openpi.models_pytorch.IntentionVLA_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.optimizer as _optimizer


def classify_loaded_weights(missing_keys, all_keys):
    missing_set = set(missing_keys)
    categories = {
        'vlm': {
            'loaded': 0,
            'total': 0
        },
        'action_expert': {
            'loaded': 0,
            'total': 0
        },
        'projection': {
            'loaded': 0,
            'total': 0
        },
        'gaze': {
            'loaded': 0,
            'total': 0
        },
    }
    
    # Define matching patterns for each module
    projection_patterns = ['action_in_proj', 'action_out_proj', 'state_proj', 'time_mlp']
    gaze_patterns = ['gaze_init_token', 'gaze_head']
    
    for key in all_keys:
        # VLM (excluding gemma_expert)
        if key.startswith('paligemma_with_expert.paligemma.') and 'gemma_expert' not in key:
            categories['vlm']['total'] += 1
            if key not in missing_set:
                categories['vlm']['loaded'] += 1
        # Action Expert
        elif key.startswith('paligemma_with_expert.gemma_expert.'):
            categories['action_expert']['total'] += 1
            if key not in missing_set:
                categories['action_expert']['loaded'] += 1
        # Projection layers
        elif any(p in key for p in projection_patterns):
            categories['projection']['total'] += 1
            if key not in missing_set:
                categories['projection']['loaded'] += 1
        # Gaze layers
        elif any(p in key for p in gaze_patterns):
            categories['gaze']['total'] += 1
            if key not in missing_set:
                categories['gaze']['loaded'] += 1
    
    return categories


def load_from_pretrain(model: torch.nn.Module, config: _config.TrainConfig, accelerator=None):
    """Load pretrained weights with shape checks for projection layers."""
    if config.pytorch_weight_path is None:
        return

    # Get actual model (remove DDP/Accelerator wrapper)
    if accelerator is not None:
        model_to_load = accelerator.unwrap_model(model)
    else:
        model_to_load = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
    model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
    ckpt_tensors = safetensors.torch.load_file(model_path, device="cpu")
    current_state = model_to_load.state_dict()

    skipped_keys = []
    for key in current_state.keys():
        if (key in ckpt_tensors) and (ckpt_tensors[key].shape != current_state[key].shape):
            skipped_keys.append(key)
            del ckpt_tensors[key]

    missing_keys, unexpected_keys = model_to_load.load_state_dict(ckpt_tensors, strict=False)
    missing_keys = [k for k in missing_keys if k not in skipped_keys]

    # Classify loaded weights by module
    load_stats = classify_loaded_weights(missing_keys, current_state.keys())
    
    logging.info("=" * 80)
    logging.info("Weight loading statistics:")
    for module_name, stats in load_stats.items():
        loaded = stats['loaded']
        total = stats['total']
        percentage = (loaded / total * 100) if total > 0 else 0
        status = "OK" if loaded == total else ("NONE" if loaded == 0 else "PARTIAL")
        logging.info(
            f"  [{status:7s}] {module_name:15s}: {loaded:4d}/{total:4d} ({percentage:5.1f}%) loaded"
        )
    logging.info("=" * 80)
    
    if skipped_keys:
        logging.info("Skipped keys due to shape mismatch (%d):", len(skipped_keys))
        for key in skipped_keys[:10]:
            logging.info(f"  - {key}")
        if len(skipped_keys) > 10:
            logging.info(f"  ... and {len(skipped_keys) - 10} more")
    
    if unexpected_keys:
        logging.info("Unexpected keys in checkpoint (%d):", len(unexpected_keys))
        for key in unexpected_keys[:10]:
            logging.info(f"  - {key}")
        if len(unexpected_keys) > 10:
            logging.info(f"  ... and {len(unexpected_keys) - 10} more")
    
    if missing_keys:
        logging.info("Missing keys in checkpoint (%d, will be randomly initialized):", len(missing_keys))
        for key in missing_keys[:10]:
            logging.info(f"  - {key}")
        if len(missing_keys) > 10:
            logging.info(f"  ... and {len(missing_keys) - 10} more")


def init_logging():
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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def create_accelerator(config):
    """
    Create accelerator instance to replace traditional DDP setup
    
    Args:
        config: TrainConfig instance
        
    Returns:
        Accelerator: Configured accelerator instance
    """
    # Set up project and logging directories
    logs_dir = config.checkpoint_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    project_config = ProjectConfiguration(
        project_dir=str(config.checkpoint_dir),
        logging_dir=str(logs_dir)
    )
    
    # DeepSpeed configuration (if config file is provided)
    deepspeed_plugin = None
    if hasattr(config, 'deepspeed_config_file') and config.deepspeed_config_file:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=config.deepspeed_config_file)
        logging.info(f"DeepSpeed enabled: {config.deepspeed_config_file}")
    
    # Mixed precision configuration (based on pytorch_training_precision)
    mixed_precision = "no"
    if config.pytorch_training_precision == "bfloat16":
        mixed_precision = "bf16"
    elif config.pytorch_training_precision == "float16":
        mixed_precision = "fp16"
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        gradient_as_bucket_view=True
    )

    # Gradient accumulation configuration
    gradient_accumulation_steps = config.gradient_accumulation_steps
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Calculate per_device_batch_size (if not specified)
    if config.per_device_batch_size is None:
        total_batch_size = config.batch_size
        world_size = accelerator.num_processes
        if total_batch_size % (world_size * gradient_accumulation_steps) != 0:
            raise ValueError(
                f"batch_size ({total_batch_size}) must be divisible by "
                f"(num_gpus × gradient_accumulation_steps) = ({world_size} × {gradient_accumulation_steps})"
            )
        # Use object.__setattr__ to modify frozen dataclass
        object.__setattr__(config, 'per_device_batch_size', 
                          total_batch_size // (world_size * gradient_accumulation_steps))
    
    # Calculate actual global batch size
    actual_global_batch_size = (
        config.per_device_batch_size * accelerator.num_processes * gradient_accumulation_steps
        if config.per_device_batch_size is not None
        else config.batch_size
    )
    
    logging.info("Accelerator initialization completed:")
    logging.info(f"   - Mixed precision: {mixed_precision}")
    logging.info(f"   - DeepSpeed: {'Enabled' if deepspeed_plugin else 'Disabled'}")
    logging.info(f"   - Device: {accelerator.device}")
    logging.info(f"   - Number of GPUs: {accelerator.num_processes}")
    logging.info(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"   - Batch size per GPU: {config.per_device_batch_size}")
    logging.info(f"   - Global batch size: {actual_global_batch_size}")
    
    return accelerator


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, accelerator=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        if accelerator is not None:
            model_to_save = accelerator.unwrap_model(model)
        else:
            model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        if accelerator is not None:
            accelerator.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
        else:
            torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, accelerator=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            # Get actual model (remove accelerator wrapper)
            if accelerator is not None:
                model_to_load = accelerator.unwrap_model(model)
            else:
                model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            
            # Load with strict=False to allow partial weight loading (for new gaze layers)
            state_dict = safetensors.torch.load_file(str(safetensors_path), device=str(device))
            missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.info(f"Missing keys (will be randomly initialized): {missing_keys}")
            if unexpected_keys:
                logging.info(f"Unexpected keys (will be ignored): {unexpected_keys}")
            
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def classify_parameters(model, accelerator):
    param_groups = {
        "vlm": [],
        "action_expert": [],
        "projection": [],
        "gaze": [],
    }
    
    # Get actual model (unwrap DDP/Accelerator wrapper)
    if hasattr(accelerator, 'unwrap_model'):
        actual_model = accelerator.unwrap_model(model)
    else:
        actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    # Classify parameters
    for name, param in actual_model.named_parameters():
        if "paligemma_with_expert.paligemma" in name and "gemma_expert" not in name:
            param_groups["vlm"].append((name, param))
        elif "paligemma_with_expert.gemma_expert" in name:
            param_groups["action_expert"].append((name, param))
        elif any(p in name for p in ["action_in_proj", "action_out_proj", "state_proj", "action_time_mlp"]):
            param_groups["projection"].append((name, param))
        elif any(p in name for p in ["gaze_init_token", "gaze_head"]):
            param_groups["gaze"].append((name, param))
    
    # Print classification statistics
    if accelerator.is_main_process:
        logging.info("=== Model parameter classification ===")
        for category, params in param_groups.items():
            num_params = sum(p.numel() for _, p in params)
            num_groups = len(params)
            logging.info(f"  - {category}: {num_groups} groups, {num_params:,} parameters")
    
    return param_groups


def update_freeze_status(model, model_config, current_step, warmup_steps, accelerator, param_groups=None):

    in_warmup = current_step < warmup_steps
    
    # Use cached parameter groups to avoid re-executing classify_parameters
    if param_groups is None:
        param_groups = classify_parameters(model, accelerator)
    
    # Get freeze configuration from model config
    freeze_config = model_config.freeze_config
    
    if accelerator.is_main_process:
        logging.info(f"=== Freeze status update (step={current_step}, in_warmup={in_warmup}) ===")
    
    # Update requires_grad for each category
    for category in param_groups.keys():
        params = param_groups[category]
        freeze_mode = freeze_config.get(category, "hot")
        
        # Determine requires_grad based on freeze mode
        if freeze_mode == "freeze":
            requires_grad = False
        elif freeze_mode == "warmup_freeze":
            requires_grad = not in_warmup
        else:  # "hot"
            requires_grad = True
        
        # Apply requires_grad setting
        for name, param in params:
            param.requires_grad = requires_grad
        
        # Log output
        if accelerator.is_main_process:
            status = "trainable" if requires_grad else "frozen"
            num_params = sum(p.numel() for _, p in params)
            logging.info(f"  - {category:15s} ({freeze_mode:13s}): {status:10s} | {len(params)} groups, {num_params:,} params")


class TimingLogger:
    """Precise timing logger for analyzing time consumption of various stages in training process"""
    
    def __init__(self, is_main=True, log_interval=10, detailed_log_interval=1):
        self.is_main = is_main
        self.log_interval = log_interval
        self.detailed_log_interval = detailed_log_interval  # Real-time detailed timing log interval
        self.timing_data = {}
        self.step_count = 0
        self.current_step_timings = {}  # Current step timing information
        
    def start_timer(self, name):
        """Start timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU operations are completed
        self.timing_data[f"{name}_start"] = time.time()
        
    def end_timer(self, name):
        """End timing and return duration"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure GPU operations are completed
        end_time = time.time()
        start_time = self.timing_data.get(f"{name}_start", end_time)
        duration = end_time - start_time
        
        # Record to cumulative statistics
        if name not in self.timing_data:
            self.timing_data[name] = []
        self.timing_data[name].append(duration)
        
        # Record current step timing
        self.current_step_timings[name] = duration
        
        return duration
        
    def log_step_timing(self, step, immediate_log=False):
        """Record timing statistics for current step"""
        self.step_count += 1
        
        # Real-time detailed timing output
        if self.is_main and (step % self.detailed_log_interval == 0) and self.current_step_timings:
            self._print_detailed_step_timing(step)
        
        # Output average time consumption at regular intervals or immediately
        if immediate_log or (self.step_count % self.log_interval == 0):
            if self.is_main:
                self._print_timing_summary(step)
                
        # Clear current step timing
        self.current_step_timings = {}
                
    def _print_timing_summary(self, step):
        """Print timing statistics summary"""
        if not self.timing_data:
            return
        
        # Define total time items and sub-step order
        total_time_keys = {"total_step", "total_initialization"}
        step_order = ["checkpoint_directory_setup", "wandb_initialization", "data_loader_initialization", 
                     "sample_data_logging", "model_configuration", "model_creation", "ddp_setup", 
                     "pretrained_weights_loading", "optimizer_initialization", "checkpoint_loading",
                     "data_loading", "data_transfer", "lr_update", "forward_pass", 
                     "backward_pass", "gradient_clipping", "optimizer_step", "gradient_cleanup"]
        
        # Collect sub-steps and total time items separately
        sub_step_summaries = []
        total_time_summaries = []
        sub_step_total = 0
        
        # Process sub-steps in order
        for name in step_order:
            if name in self.timing_data and self.timing_data[name]:
                times = self.timing_data[name]
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                sub_step_total += latest_time
                sub_step_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        # Handle other unlisted sub-steps
        for name, times in self.timing_data.items():
            if (name not in step_order and name not in total_time_keys and 
                not name.endswith('_start') and times):
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                sub_step_total += latest_time
                sub_step_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
        
        # Handle total time items
        for name in total_time_keys:
            if name in self.timing_data and self.timing_data[name]:
                times = self.timing_data[name]
                avg_time = sum(times) / len(times)
                latest_time = times[-1] if times else 0
                total_time_summaries.append(f"{name}: {latest_time*1000:.2f}ms (avg: {avg_time*1000:.2f}ms)")
            
        if sub_step_summaries or total_time_summaries:
            logging.info(f"Step {step} Timing Details:")
            for summary in sub_step_summaries:
                logging.info(f"  ├─ {summary}")
            if total_time_summaries:
                logging.info(f"  ├─ [Total Time]")
                for summary in total_time_summaries:
                    logging.info(f"  ├─ {summary}")
            logging.info(f"  └─ Sub-steps Total: {sub_step_total*1000:.2f}ms")
            
    def reset_stats(self):
        """Reset statistics data"""
        for key in list(self.timing_data.keys()):
            if not key.endswith('_start'):
                self.timing_data[key] = []
                
    def _print_detailed_step_timing(self, step):
        """Print detailed timing information for a single step"""
        if not self.current_step_timings:
            return
        
        # Define total time items, these should not be included in the denominator of percentage calculation
        total_time_keys = {"total_step", "total_initialization"}
        
        # Calculate total time of actual sub-steps (excluding total time items)
        sub_step_total = sum(time_val for name, time_val in self.current_step_timings.items() 
                           if name not in total_time_keys)
        
        # Arrange timing items in execution order
        timing_order = ["data_loading", "data_transfer", "lr_update", "forward_pass", 
                       "backward_pass", "gradient_clipping", "optimizer_step", 
                       "gradient_cleanup"]
        
        timing_info = []
        
        # First process ordered sub-steps
        for name in timing_order:
            if name in self.current_step_timings:
                time_ms = self.current_step_timings[name] * 1000
                percentage = (self.current_step_timings[name] / sub_step_total) * 100 if sub_step_total > 0 else 0
                timing_info.append(f"{name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        # Add other sub-steps (excluding total time items)
        for name, time_val in self.current_step_timings.items():
            if name not in timing_order and name not in total_time_keys:
                time_ms = time_val * 1000
                percentage = (time_val / sub_step_total) * 100 if sub_step_total > 0 else 0
                timing_info.append(f"{name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        if timing_info:
            logging.info(f"Step {step} Real-time Timing: {', '.join(timing_info)}")


def train_loop(config: _config.TrainConfig):
    # Create accelerator to replace traditional DDP setup
    accelerator = create_accelerator(config)
    
    # Use accelerator's seed setting (ensures distributed consistency)
    transformers_set_seed(config.seed)
    accelerate_set_seed(config.seed, device_specific=True)
    
    is_main = accelerator.is_main_process

    # Initialize timing logger for initialization phase
    init_timer = TimingLogger(is_main=is_main, log_interval=1, detailed_log_interval=1)
    
    if is_main:
        logging.info("Starting training initialization, detailed timing as follows:")
    
    init_timer.start_timer("total_initialization")

    # Initialize checkpoint directory and wandb
    init_timer.start_timer("checkpoint_directory_setup")
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
    
    init_timer.end_timer("checkpoint_directory_setup")

    # Initialize wandb (only on main process)
    init_timer.start_timer("wandb_initialization")
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    init_timer.end_timer("wandb_initialization")

    # Build data loader using the unified data loader
    init_timer.start_timer("data_loader_initialization")
    # Calculate per-device batch size
    per_device_bs = config.per_device_batch_size
    gradient_accum_steps = config.gradient_accumulation_steps
    
    logging.info(f"Batch Size configuration:")
    logging.info(f"   - Batch size per GPU: {per_device_bs}")
    logging.info(f"   - Number of GPUs: {accelerator.num_processes}")
    logging.info(f"   - Gradient accumulation steps: {gradient_accum_steps}")
    logging.info(f"   - Actual global batch size: {per_device_bs * accelerator.num_processes * gradient_accum_steps}")


    # Temporarily modify config.batch_size to per_device_batch_size for DataLoader
    # Accelerator's prepare() will handle multi-GPU data distribution
    original_batch_size = config.batch_size
    if per_device_bs is not None:
        object.__setattr__(config, 'batch_size', per_device_bs)
    loader, data_config = build_datasets(config)
    object.__setattr__(config, 'batch_size', original_batch_size)  # Restore original value
    init_timer.end_timer("data_loader_initialization")

    # Log sample images to wandb on first batch
    init_timer.start_timer("sample_data_logging")
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")
    init_timer.end_timer("sample_data_logging")

    # Build model
    init_timer.start_timer("model_configuration")
    if not isinstance(config.model, openpi.models.IntentionVLA_config.IntentionVLAConfig):
        # Convert dataclass to IntentionVLAConfig if needed
        model_cfg = openpi.models.IntentionVLA_config.IntentionVLAConfig(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=config.model.paligemma_variant,
            action_expert_variant=config.model.action_expert_variant,
            pi05=config.model.pi05,
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    init_timer.end_timer("model_configuration")

    init_timer.start_timer("model_creation")
    model = openpi.models_pytorch.IntentionVLA_pytorch.IntentionVLA_pytorch(model_cfg).to(accelerator.device)
    init_timer.end_timer("model_creation")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(accelerator.device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if accelerator.num_processes >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from pretrain if specified (for fine-tuning)
    # Note: Load before accelerator.prepare() to avoid wrapper issues
    init_timer.start_timer("pretrained_weights_loading")
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        load_from_pretrain(model, config, accelerator=None)  # Pass None since not yet wrapped
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    init_timer.end_timer("pretrained_weights_loading")

    # Gradient checkpointing configuration
    enable_gradient_checkpointing = config.enable_gradient_checkpointing
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        model.gradient_checkpointing_disable()
        logging.info("Gradient checkpointing is disabled")

    init_timer.start_timer("freeze_initialization")
    warmup_steps_for_freeze = config.lr_schedule.warmup_steps
    param_groups_cache = classify_parameters(model, accelerator)
    update_freeze_status(
        model=model,
        model_config=model_cfg,
        current_step=0,
        warmup_steps=warmup_steps_for_freeze, 
        accelerator=accelerator,
        param_groups=param_groups_cache
    )
    init_timer.end_timer("freeze_initialization")

    # Optimizer + learning rate schedule from config
    init_timer.start_timer("optimizer_initialization")
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    init_timer.end_timer("optimizer_initialization")

    # Load checkpoint if resuming (before accelerator.prepare())
    init_timer.start_timer("checkpoint_loading")
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, accelerator.device, accelerator=None)
        logging.info(f"Resumed training from step {global_step}")
    init_timer.end_timer("checkpoint_loading")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Prepare model, optimizer, and data loader with accelerator
    # This must be done after model/optimizer creation but before training
    init_timer.start_timer("accelerator_preparation")
    model, optim, loader = accelerator.prepare(model, optim, loader)
    logging.info("Accelerator preparation completed")
    init_timer.end_timer("accelerator_preparation")

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={accelerator.num_processes}")
        logging.info(f"Training config: batch_size={config.batch_size}, per_device_batch_size={config.per_device_batch_size}, num_train_steps={config.num_train_steps}")
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
        logging.info(f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
    
    # End total initialization timing
    init_timer.end_timer("total_initialization")
    
    # Output detailed timing summary for initialization phase
    if is_main:
        init_timer.log_step_timing(0, immediate_log=True)
        logging.info("Training initialization completed, starting training loop")

    # Initialize timing logger - show detailed timing for each step
    timing_logger = TimingLogger(is_main=is_main, log_interval=config.log_interval, detailed_log_interval=1)
    
    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Data loader iteration timing
        timing_logger.start_timer("data_loading")
        for observation, actions in loader:
            timing_logger.end_timer("data_loading")
            
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            if global_step == warmup_steps_for_freeze and global_step > 0:
                if is_main:
                    logging.info(f"Warmup completed at step {global_step}, updating freeze status...")
                
                # Update freeze status (will change requires_grad for warmup_freeze parameters)
                update_freeze_status(
                    model=model,
                    model_config=model_cfg,
                    current_step=global_step,
                    warmup_steps=warmup_steps_for_freeze,
                    accelerator=accelerator,
                    param_groups=param_groups_cache
                )
                
            # Data transfer to GPU timing
            timing_logger.start_timer("data_transfer")
            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(
                lambda x: x.to(accelerator.device) if isinstance(x, torch.Tensor) else x, 
                observation
            )
            actions = actions.to(torch.float32)
            actions = actions.to(accelerator.device)
            timing_logger.end_timer("data_transfer")

            # Wrap training step with accelerator.accumulate() for gradient accumulation
            with accelerator.accumulate(model):
                # Start timing for entire training step
                timing_logger.start_timer("total_step")
                
                # Learning rate update timing
                timing_logger.start_timer("lr_update")
                # Update LR
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step)
                timing_logger.end_timer("lr_update")

                # Forward pass timing
                timing_logger.start_timer("forward_pass")
                loss_dict = model(observation, actions)

                # Extract loss components
                gaze_cross_entropy_loss = loss_dict['gaze_cross_entropy_loss']
                gaze_mse_loss = loss_dict['gaze_mse_loss']
                action_loss = loss_dict['action_loss']
                loss = loss_dict['total_loss']

                timing_logger.end_timer("forward_pass")

                # Backward pass timing - use accelerator.backward()
                timing_logger.start_timer("backward_pass")
                accelerator.backward(loss)
                timing_logger.end_timer("backward_pass")

                # Log memory usage after backward pass
                if global_step < 5 and accelerator.is_main_process and torch.cuda.is_available():
                    log_memory_usage(accelerator.device, global_step, "after_backward")

                # Gradient clipping timing - only when syncing gradients
                timing_logger.start_timer("gradient_clipping")
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.optimizer.clip_gradient_norm
                    )
                else:
                    grad_norm = 0.0
                timing_logger.end_timer("gradient_clipping")

                # Optimizer step timing
                timing_logger.start_timer("optimizer_step")
                optim.step()
                optim.zero_grad(set_to_none=True)
                timing_logger.end_timer("optimizer_step")
                
                # End timing for entire step
                timing_logger.end_timer("total_step")

            # Collect stats (only on main process)
            if accelerator.is_main_process:
                info_dict = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "gaze_cross_entropy_loss": gaze_cross_entropy_loss.item() if isinstance(gaze_cross_entropy_loss, torch.Tensor) else gaze_cross_entropy_loss,
                    "gaze_mse_loss": gaze_mse_loss.item() if isinstance(gaze_mse_loss, torch.Tensor) else gaze_mse_loss,
                    "action_loss": action_loss.item() if isinstance(action_loss, torch.Tensor) else action_loss,
                }
                # Only record grad_norm when syncing (avoid recording 0 values)
                if accelerator.sync_gradients:
                    info_dict["grad_norm"] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
                infos.append(info_dict)

            # Logging (only when gradients sync)
            if accelerator.sync_gradients and accelerator.is_main_process and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                
                # Calculate average gaze losses and action_loss
                avg_gaze_cross_entropy_loss = None
                if any("gaze_cross_entropy_loss" in info for info in infos):
                    vals = [info["gaze_cross_entropy_loss"] for info in infos if "gaze_cross_entropy_loss" in info and info["gaze_cross_entropy_loss"] is not None]
                    if len(vals) > 0:
                        avg_gaze_cross_entropy_loss = sum(vals) / len(vals)
                
                avg_gaze_mse_loss = None
                if any("gaze_mse_loss" in info for info in infos):
                    vals = [info["gaze_mse_loss"] for info in infos if "gaze_mse_loss" in info and info["gaze_mse_loss"] is not None]
                    if len(vals) > 0:
                        avg_gaze_mse_loss = sum(vals) / len(vals)
                
                avg_action_loss = None
                if any("action_loss" in info for info in infos):
                    vals = [info["action_loss"] for info in infos if "action_loss" in info and info["action_loss"] is not None]
                    if len(vals) > 0:
                        avg_action_loss = sum(vals) / len(vals)
                
                # Build log string
                log_parts = [
                    f"step={global_step}",
                    f"loss={avg_loss:.4f}",
                ]
                
                # Add gaze losses and action_loss
                if avg_gaze_cross_entropy_loss is not None:
                    log_parts.append(f"gaze_ce={avg_gaze_cross_entropy_loss:.4f}")
                if avg_gaze_mse_loss is not None:
                    log_parts.append(f"gaze_mse={avg_gaze_mse_loss:.4f}")
                if avg_action_loss is not None:
                    log_parts.append(f"action_loss={avg_action_loss:.4f}")
                
                # Add learning rate and gradient norm
                log_parts.append(f"lr={avg_lr:.2e}")
                if avg_grad_norm is not None:
                    log_parts.append(f"grad_norm={avg_grad_norm:.2f}")
                log_parts.append(f"time={elapsed:.1f}s")
                
                logging.info(" ".join(log_parts))

                # Record timing information
                timing_logger.log_step_timing(global_step)

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    
                    # Add gaze losses and action_loss to wandb
                    if avg_gaze_cross_entropy_loss is not None:
                        log_payload["gaze_cross_entropy_loss"] = avg_gaze_cross_entropy_loss
                    if avg_gaze_mse_loss is not None:
                        log_payload["gaze_mse_loss"] = avg_gaze_mse_loss
                    if avg_action_loss is not None:
                        log_payload["action_loss"] = avg_action_loss
                    
                    # Add detailed timing information to wandb
                    for name, times in timing_logger.timing_data.items():
                        if not name.endswith('_start') and times:
                            avg_time_ms = (sum(times) / len(times)) * 1000  # Convert to milliseconds
                            log_payload[f"timing/{name}_ms"] = avg_time_ms
                    
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            # Update global_step only when gradients sync
            if accelerator.sync_gradients:
                global_step += 1
                
                # Save checkpoint (pass accelerator parameter)
                save_checkpoint(model, optim, global_step, config, accelerator.is_main_process, data_config, accelerator)
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    postfix_dict = {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step
                    }
                    
                    # Add gaze losses and action_loss to progress bar
                    if gaze_cross_entropy_loss is not None:
                        gaze_ce_val = gaze_cross_entropy_loss.item() if isinstance(gaze_cross_entropy_loss, torch.Tensor) else gaze_cross_entropy_loss
                        postfix_dict["g_ce"] = f"{gaze_ce_val:.4f}"
                    if gaze_mse_loss is not None:
                        gaze_mse_val = gaze_mse_loss.item() if isinstance(gaze_mse_loss, torch.Tensor) else gaze_mse_loss
                        postfix_dict["g_mse"] = f"{gaze_mse_val:.4f}"
                    if action_loss is not None:
                        action_loss_val = action_loss.item() if isinstance(action_loss, torch.Tensor) else action_loss
                        postfix_dict["a_loss"] = f"{action_loss_val:.4f}"
                    
                    pbar.set_postfix(postfix_dict)
            
            # Start timing for next data loading
            if global_step < config.num_train_steps - 1:
                timing_logger.start_timer("data_loading")

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if accelerator.is_main_process and config.wandb_enabled:
        wandb.finish()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
