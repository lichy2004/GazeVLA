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

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


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
            if accelerator is not None:
                model_to_load = accelerator.unwrap_model(model)
            else:
                model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
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


def train_loop(config: _config.TrainConfig):
    # Create accelerator to replace traditional DDP setup
    accelerator = create_accelerator(config)
    
    # Use accelerator's seed setting
    transformers_set_seed(config.seed)
    accelerate_set_seed(config.seed, device_specific=True)
    
    is_main = accelerator.is_main_process

    # Initialize checkpoint directory and wandb
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

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
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

    # Log sample images to wandb on first batch
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

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
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

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(accelerator.device)

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
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Gradient checkpointing configuration
    enable_gradient_checkpointing = config.enable_gradient_checkpointing
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        model.gradient_checkpointing_disable()
        logging.info("Gradient checkpointing is disabled")

    # Optimizer + learning rate schedule from config
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

    # Load checkpoint if resuming (before accelerator.prepare())
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, accelerator.device, accelerator=None)
        logging.info(f"Resumed training from step {global_step}")

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
    model, optim, loader = accelerator.prepare(model, optim, loader)
    logging.info("Accelerator preparation completed")

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
    
    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(
                lambda x: x.to(accelerator.device) if isinstance(x, torch.Tensor) else x, 
                observation
            )
            actions = actions.to(torch.float32)
            actions = actions.to(accelerator.device)

            # Wrap training step with accelerator.accumulate() for gradient accumulation
            with accelerator.accumulate(model):
                # Update LR
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step)

                # Forward pass
                losses = model(observation, actions)
                # Ensure losses is a tensor and handle different return types
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=accelerator.device, dtype=torch.float32)

                loss = losses.mean()

                # Backward pass - use accelerator.backward()
                accelerator.backward(loss)

                # Log memory usage after backward pass
                if global_step < 5 and accelerator.is_main_process and torch.cuda.is_available():
                    log_memory_usage(accelerator.device, global_step, "after_backward")

                # Gradient clipping - only when syncing gradients
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.optimizer.clip_gradient_norm
                    )
                else:
                    grad_norm = 0.0

                # Optimizer step
                optim.step()
                optim.zero_grad(set_to_none=True)

            # Collect stats (only on main process)
            if accelerator.is_main_process:
                info_dict = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
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
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

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
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                    )

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