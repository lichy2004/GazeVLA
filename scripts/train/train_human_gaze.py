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

import openpi.models.IntentionVLA_config
import openpi.models_pytorch.IntentionVLA_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.optimizer as _optimizer


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


def load_from_pretrain(model: torch.nn.Module, config: _config.TrainConfig):
    """Load pretrained weights with shape checks for projection layers."""
    if config.pytorch_weight_path is None:
        return

    model_to_load = (
        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    )
    model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
    ckpt_tensors = safetensors.torch.load_file(model_path, device="cpu")
    current_state = model_to_load.state_dict()

    projection_bases = ["action_in_proj", "action_out_proj", "state_proj", "gaze_head"]
    skipped_keys = []
    for base in projection_bases:
        for suffix in ("weight", "bias"):
            key = f"{base}.{suffix}"
            if key in ckpt_tensors:
                if ckpt_tensors[key].shape != current_state[key].shape:
                    skipped_keys.append(key)
                    del ckpt_tensors[key]

    missing_keys, unexpected_keys = model_to_load.load_state_dict(ckpt_tensors, strict=False)


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


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


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


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
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
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
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


def load_checkpoint(model, optimizer, checkpoint_dir, device):
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
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            
            # 使用 strict=False 允许加载部分权重（忽略新增的 gaze CoT 层）
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
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize timing logger for initialization phase
    init_timer = TimingLogger(is_main=is_main, log_interval=1, detailed_log_interval=1)
    
    if is_main:
        logging.info("🚀 Starting training initialization, detailed timing as follows:")
    
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
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
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
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    init_timer.end_timer("model_configuration")

    init_timer.start_timer("model_creation")
    model = openpi.models_pytorch.IntentionVLA_pytorch.IntentionVLA_pytorch(model_cfg).to(device)
    init_timer.end_timer("model_creation")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    init_timer.start_timer("ddp_setup")
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )
    init_timer.end_timer("ddp_setup")

    # Load weights from pretrain if specified (for fine-tuning)
    init_timer.start_timer("pretrained_weights_loading")
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        load_from_pretrain(model, config)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    init_timer.end_timer("pretrained_weights_loading")

    # Optimizer + learning rate schedule from config
    init_timer.start_timer("optimizer_initialization")
    if isinstance(config.lr_schedule, _optimizer.ConstantSchedule):
        const_lr = config.lr_schedule.lr
        warmup_steps = 0
        peak_lr = const_lr
        decay_steps = 0
        end_lr = const_lr
        lr_is_constant = True
    else:
        warmup_steps = config.lr_schedule.warmup_steps
        peak_lr = config.lr_schedule.peak_lr
        decay_steps = config.lr_schedule.decay_steps
        end_lr = config.lr_schedule.decay_lr
        lr_is_constant = False

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    init_timer.end_timer("optimizer_initialization")

    # Load checkpoint if resuming
    init_timer.start_timer("checkpoint_loading")
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")
    init_timer.end_timer("checkpoint_loading")

    def lr_schedule(step: int):
        if lr_is_constant:
            return peak_lr
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        if lr_is_constant:
            logging.info(
                f"LR schedule: constant lr={peak_lr:.2e}"
            )
        else:
            logging.info(
                f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
            )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
    
    # End total initialization timing
    init_timer.end_timer("total_initialization")
    
    # Output detailed timing summary for initialization phase
    if is_main:
        init_timer.log_step_timing(0, immediate_log=True)
        logging.info("✅ Training initialization completed, starting training loop")

    # Initialize timing logger - show detailed timing for each step
    timing_logger = TimingLogger(is_main=is_main, log_interval=config.log_interval, detailed_log_interval=1)
    
    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # Data loader iteration timing
        timing_logger.start_timer("data_loading")
        for observation, actions in loader:
            timing_logger.end_timer("data_loading")
            # Start timing for entire training step
            timing_logger.start_timer("total_step")
            
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # Data transfer to GPU timing
            timing_logger.start_timer("data_transfer")
            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            timing_logger.end_timer("data_transfer")

            # Learning rate update timing
            timing_logger.start_timer("lr_update")
            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)
            timing_logger.end_timer("lr_update")

            # Forward pass timing
            timing_logger.start_timer("forward_pass")
            loss_dict = model(observation, actions)

            # 处理不同的返回类型
            gaze_loss = loss_dict['gaze_loss']
            action_loss = loss_dict['action_loss']
            loss = loss_dict['total_loss']

            timing_logger.end_timer("forward_pass")

            # Backward pass timing
            timing_logger.start_timer("backward_pass")
            loss.backward()
            timing_logger.end_timer("backward_pass")

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping timing
            timing_logger.start_timer("gradient_clipping")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            timing_logger.end_timer("gradient_clipping")

            # Optimizer step timing
            timing_logger.start_timer("optimizer_step")
            optim.step()
            optim.zero_grad(set_to_none=True)
            timing_logger.end_timer("optimizer_step")

            # Gradient cleanup timing
            timing_logger.start_timer("gradient_cleanup")
            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None
            timing_logger.end_timer("gradient_cleanup")
            
            # End timing for entire step
            timing_logger.end_timer("total_step")

            # Collect stats
            if is_main:
                info_dict = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "gaze_loss": gaze_loss.item() if isinstance(gaze_loss, torch.Tensor) else gaze_loss,
                    "action_loss": action_loss.item() if isinstance(action_loss, torch.Tensor) else action_loss,
                }
                infos.append(info_dict)

            if is_main and (global_step % config.log_interval == 0):
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
                
                # 计算平均 gaze_loss 和 action_loss（如果存在）
                avg_gaze_loss = None
                if any("gaze_loss" in info for info in infos):
                    vals = [info["gaze_loss"] for info in infos if "gaze_loss" in info and info["gaze_loss"] is not None]
                    if len(vals) > 0:
                        avg_gaze_loss = sum(vals) / len(vals)
                
                avg_action_loss = None
                if any("action_loss" in info for info in infos):
                    vals = [info["action_loss"] for info in infos if "action_loss" in info and info["action_loss"] is not None]
                    if len(vals) > 0:
                        avg_action_loss = sum(vals) / len(vals)
                
                # 构建日志字符串
                log_parts = [
                    f"step={global_step}",
                    f"loss={avg_loss:.4f}",
                ]
                
                # 添加 gaze_loss 和 action_loss（如果存在）
                if avg_gaze_loss is not None:
                    log_parts.append(f"gaze_loss={avg_gaze_loss:.4f}")
                if avg_action_loss is not None:
                    log_parts.append(f"action_loss={avg_action_loss:.4f}")
                
                # 添加学习率和梯度范数
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
                    
                    # 添加 gaze_loss 和 action_loss 到 wandb（如果存在）
                    if avg_gaze_loss is not None:
                        log_payload["gaze_loss"] = avg_gaze_loss
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

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                postfix_dict = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    "step": global_step
                }
                
                # 添加 gaze_loss 和 action_loss 到进度条（如果存在）
                if gaze_loss is not None:
                    gaze_loss_val = gaze_loss.item() if isinstance(gaze_loss, torch.Tensor) else gaze_loss
                    postfix_dict["g_loss"] = f"{gaze_loss_val:.4f}"
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
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
