import logging
import math
from tkinter.constants import FALSE
from typing import Tuple, Optional, Any, List

import os
import sys
import cv2
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_human as _preprocessing
from openpi.models_pytorch.VLIA_gemma import PaliGemmaWithExpertModel
from openpi.models_pytorch.gaze_tokenizer import GazeTokenizer
from openpi.utils.dataset_utils import generate_heatmap, visualize_heatmap, transform_images


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class VLIA_pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        # Gaze COT
        paligemma_vocab_size = _gemma.PALIGEMMA_VOCAB_SIZE
        paligemma_hidden_size = paligemma_config.width
        self.gaze_tokenizer = GazeTokenizer(
            num_bins=config.gaze_vocab_size,
            image_size=224,
            paligemma_vocab_size=paligemma_vocab_size,
        )
        self.gaze_init_token = nn.Parameter(torch.randn(1, paligemma_hidden_size))  # [1, 2048]
        self.gaze_head = nn.Linear(paligemma_hidden_size, config.gaze_vocab_size)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for VLIA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train, image_keys=self.config.image_keys
        )
        if not self.config.gaze_cot:
            observation.gaze = None
            observation.gaze_mask = None
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.gaze,
            observation.action_mask,
            observation.gaze_mask,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, gaze=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # images: list([bs, 3, h, w])
        # img_masks: list([bs])
        # lang_tokens: [bs, 48]
        # lang_masks: [bs, 48]
        # gaze: [bs, 2]

        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)                # [bs, 256, 2048]
            pad_masks.append(img_mask)          # [bs, 256]
            att_masks += [0] * num_img_embs     # +[256]

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        num_lang_embs = lang_emb.shape[1]
        embs.append(lang_emb)               # [bs, 48, 2048]
        pad_masks.append(lang_masks)        # [bs, 48]
        att_masks += [0] * num_lang_embs    # +[48]

        # Process gaze tokens
        def gaze_embed_func(gaze):
            gaze_tokens = self.gaze_tokenizer.encode(gaze)
            gaze_emb = self.paligemma_with_expert.embed_language_tokens(gaze_tokens)
            gaze_emb_dim = gaze_emb.shape[-1]
            return gaze_emb * math.sqrt(gaze_emb_dim)
        
        if gaze is not None:
            gaze_init_emb = self.gaze_init_token.unsqueeze(0).expand(bsize, -1, -1) # [bs, 1, 2048]
            gaze_emb = self._apply_checkpoint(gaze_embed_func, gaze)                # [bs, 2, 2048]
            gaze_emb = torch.cat([gaze_init_emb, gaze_emb], dim=1)                  # [bs, 3, 2048]
            gaze_masks = torch.ones((bsize, 3), dtype=torch.bool, device=gaze_emb.device)
            bsize, num_gaze_embs = gaze_emb.shape[:2]
            embs.append(gaze_emb)
            pad_masks.append(gaze_masks)
            att_masks += [1] * num_gaze_embs

        # concat all embeddings
        embs = torch.cat(embs, dim=1)               # [bs, L=256+48+3, 2048]
        pad_masks = torch.cat(pad_masks, dim=1)     # [bs, L]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)  # [L]

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))    # [bs, L]

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32) # [bs, dim]

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)[:, None, :]  # [bs, 1, 1024]

            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device) # [bs, 1]
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)  # [bs, 1024]

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)    # [bs, frame, 1024]

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)       # [bs, frame, 1024]
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb) # [bs, frame, 1024]
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)           # [bs, 1+frame, 1024]
        pad_masks = torch.cat(pad_masks, dim=1) # [bs, 1+frame]
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))    # [bs, 1+frame]

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None):
        """Run the training forward pass and compute the losses."""
        images, img_masks, lang_tokens, lang_masks, state, gaze, action_mask, gaze_mask \
            = self._preprocess_observation(observation, train=True)
        # images: list([bs, c, h, w]) 
        # img_masks: list([bs]) 
        # lang_tokens: [bs, 48] 
        # lang_masks: [bs, 48] 
        # state: [bs, dim]
        # gaze: [bs, 2]
        # action_mask: [bs, T, 2] or [bs, T, 1]
        # gaze_mask: [bs, 1]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)    # [bs, frame, dim]

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)   # [bs]

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions     # [bs, frame, dim]
        u_t = noise - actions                                           # [bs, frame, dim]

        # prefix and suffix embedding
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, gaze)
        # prefix_embs: [bs, 307, 2048] = [image(256), text(48), gaze_init(1), gaze(2)]
        # prefix_pad_masks: [bs, 307]
        # prefix_att_masks: [bs, 307]
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time)
        # suffix_embs: [bs, 1+frame, 1024]
        # suffix_pad_masks: [bs, 1+frame]
        # suffix_att_masks: [bs, 1+frame]
        # adarms_cond: None

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1) # [bs, L]
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1) # [bs, L]

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)  # [bs, L, L]
        position_ids = torch.cumsum(pad_masks, dim=1) - 1       # [bs, L]

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)    # [bs, 1, L, L]

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(    # [bs, 307, 2048], [bs, 1+T, 1024]
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # ============================== action Loss ==============================
        suffix_out = suffix_out[:, -self.config.action_horizon :]   # [bs, T, 1024]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)  # [bs, T, action_dim]
        
        action_dim = self.config.action_dim
        if action_mask.shape[-1] == 2:
            # human data
            action_mask_left = action_mask[:, :, 0:1].expand(-1, -1, action_dim // 2)       # [bs, T, 24]
            action_mask_right = action_mask[:, :, 1:2].expand(-1, -1, action_dim // 2)      # [bs, T, 24]
            action_mask_expanded = torch.cat([action_mask_left, action_mask_right], dim=2)  # [bs, T, action_dim]
        else:
            # robot data: mask only valid dimensions, zero-padded dimensions should not contribute to loss
            data_action_dim = self.config.data_action_dim if self.config.data_action_dim else action_dim
            valid_mask = action_mask.expand(-1, -1, data_action_dim)            # [bs, T, data_action_dim]
            if data_action_dim < action_dim:
                padding = torch.zeros(action_mask.shape[0], action_mask.shape[1], action_dim - data_action_dim,
                                      dtype=valid_mask.dtype, device=valid_mask.device)
                action_mask_expanded = torch.cat([valid_mask, padding], dim=2)  # [bs, T, action_dim]
            else:
                action_mask_expanded = valid_mask
        action_loss_per_element = F.mse_loss(u_t, v_t, reduction="none")        # [bs, T, action_dim]
        action_loss_per_element = action_loss_per_element * action_mask_expanded.float()
        num_valid = action_mask_expanded.float().sum()
        action_loss = action_loss_per_element.sum() / (num_valid + 1e-8)
        action_loss *= self.config.action_loss_weight

        # ============================== gaze Loss (token) ==============================
        gaze_cross_entropy_loss = torch.tensor(0.0, device=prefix_out.device)
        gaze_mse_loss = torch.tensor(0.0, device=prefix_out.device)
        
        if self.config.gaze_cot and gaze is not None:
            # cross entropy loss
            num_image_tokens = self.config.num_image_tokens
            num_text_tokens = self.config.num_text_tokens
            num_gaze_tokens = self.config.num_gaze_tokens
            gaze_init_token_idx = num_image_tokens + num_text_tokens
            
            gaze_hidden = prefix_out[:, gaze_init_token_idx:gaze_init_token_idx+num_gaze_tokens, :]   # [bs, 2, 2048]
            logits = self.gaze_head(gaze_hidden.float())        # [bs, 2, 224]
            
            gaze_gt = gaze
            gaze_tokens_gt = self.gaze_tokenizer.encode(gaze_gt)   # [bs, 2]，token IDs range: [256928, 257151]
            vocab_offset = self.gaze_tokenizer.vocab_offset
            gaze_labels_gt = gaze_tokens_gt - vocab_offset      # [bs, 2]，gaze token [256928, 257151] -> [0, 223]
            gaze_labels_gt = torch.clamp(gaze_labels_gt, 0, self.config.gaze_vocab_size - 1)
            
            # Compute per-token cross entropy loss
            gaze_mask_expanded = gaze_mask.expand(-1, 2).float()  # [bs, 2]
            gaze_loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [bs*2, 224]
                gaze_labels_gt.view(-1),           # [bs*2]
                reduction="none",                  # [bs*2]
            )
            gaze_loss_per_token = gaze_loss_per_token.view(gaze_labels_gt.shape)  # [bs, 2]
            gaze_loss_per_token = gaze_loss_per_token * gaze_mask_expanded
            num_valid = gaze_mask_expanded.sum()
            gaze_cross_entropy_loss = gaze_loss_per_token.sum() / (num_valid + 1e-8)
            gaze_cross_entropy_loss *= self.config.gaze_loss_weight

            # mse loss (for monitoring only, not used in backprop)
            gaze_token_pred = torch.argmax(logits, dim=-1)                                  # [bs, 2]
            gaze_token_pred = gaze_token_pred + self.gaze_tokenizer.vocab_offset            # [bs, 2]
            gaze_pred = self.gaze_tokenizer.decode(gaze_token_pred)                         # [bs, 2]
            gaze_loss_mse_per_element = F.mse_loss(gaze_gt, gaze_pred, reduction="none")    # [bs, 2]
            gaze_loss_mse_per_element = gaze_loss_mse_per_element * gaze_mask_expanded
            gaze_mse_loss = gaze_loss_mse_per_element.sum() / (num_valid + 1e-8)


        # ===== return loss =====
        total_loss = action_loss + gaze_cross_entropy_loss
        return {
            'gaze_cross_entropy_loss': gaze_cross_entropy_loss,
            'gaze_mse_loss': gaze_mse_loss,
            'action_loss': action_loss,
            'total_loss': total_loss
        }


    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Inference forward pass to compute actions.
        
        Supports two modes based on self.config.gaze_cot:
        - gaze_cot=True: Chain-of-Thought inference (image+text -> gaze -> action)
        - gaze_cot=False: Direct inference (image+text -> action)
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, gaze, action_mask, gaze_mask \
            = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        if self.config.gaze_cot:
            # ===== CoT Mode: Predict gaze first, then action =====
            # Phase 1: Autoregressively predict gaze tokens
            # Compute KV cache for image+text
            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            
            # Autoregressively predict gaze tokens
            predicted_gaze_token_ids = []
            current_prefix_len = prefix_pad_masks.shape[1]
            gaze_token_num = 3
            
            for i in range(self.config.num_gaze_tokens):
                if i == 0:
                    # First iteration: add gaze_init_token
                    gaze_init_emb = self.gaze_init_token.unsqueeze(0).expand(bsize, -1, -1)
                    gaze_init_emb = gaze_init_emb * math.sqrt(gaze_token_num)
                    
                    new_position_ids = torch.full((bsize, 1), current_prefix_len, dtype=torch.long, device=device)
                    new_att_mask = torch.zeros(
                        bsize, 1, 1, current_prefix_len + 1,
                        dtype=prefix_att_2d_masks_4d.dtype,
                        device=device
                    )
                    
                    (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                        attention_mask=new_att_mask,
                        position_ids=new_position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=[gaze_init_emb, None],
                        use_cache=True,
                    )
                else:
                    # Subsequent iterations: embed previous predicted token
                    prev_token_emb = self.paligemma_with_expert.embed_language_tokens(
                        predicted_gaze_token_ids[-1].unsqueeze(1)
                    )
                    prev_token_emb = prev_token_emb * math.sqrt(gaze_token_num)

                    new_position_ids = torch.full((bsize, 1), current_prefix_len + i, dtype=torch.long, device=device)
                    new_att_mask = torch.zeros(
                        bsize, 1, 1, current_prefix_len + i + 1,
                        dtype=prefix_att_2d_masks_4d.dtype,
                        device=device
                    )
                    
                    (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                        attention_mask=new_att_mask,
                        position_ids=new_position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=[prev_token_emb, None],
                        use_cache=True,
                    )
                
                # Predict current gaze token
                last_hidden_state = prefix_out[:, -1, :]
                gaze_logits = self.gaze_head(last_hidden_state.float())
                predicted_bin_id = torch.argmax(gaze_logits, dim=-1)
                predicted_token_id = predicted_bin_id + self.gaze_tokenizer.vocab_offset
                predicted_gaze_token_ids.append(predicted_token_id)
            
            # Forward the last predicted gaze token to update KV cache
            last_token_emb = self.paligemma_with_expert.embed_language_tokens(
                predicted_gaze_token_ids[-1].unsqueeze(1)
            )
            last_token_emb = last_token_emb * math.sqrt(gaze_token_num)
            
            final_position_ids = torch.full((bsize, 1), current_prefix_len + self.config.num_gaze_tokens, dtype=torch.long, device=device)
            final_att_mask = torch.zeros(
                bsize, 1, 1, current_prefix_len + self.config.num_gaze_tokens + 1,
                dtype=prefix_att_2d_masks_4d.dtype,
                device=device
            )
            
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=final_att_mask,
                position_ids=final_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[last_token_emb, None],
                use_cache=True,
            )
            
            # Decode predicted gaze tokens to pixel coordinates
            gaze_token_pred = torch.stack(predicted_gaze_token_ids, dim=1)
            gaze_norm_pred = self.gaze_tokenizer.decode(gaze_token_pred)
            gaze_pixel_pred = (gaze_norm_pred * 224).to(torch.int32)

            # Build prefix_pad_masks including gaze tokens
            gaze_masks_pred = torch.ones(bsize, self.config.num_gaze_tokens + 1, dtype=torch.bool, device=device)
            prefix_pad_masks_with_gaze = torch.cat([prefix_pad_masks, gaze_masks_pred], dim=1)

            # Phase 2: Predict action using gaze-conditioned KV cache
            past_key_values_final = past_key_values
            prefix_pad_masks_final = prefix_pad_masks_with_gaze
        else:
            # ===== Direct Mode: Predict action without gaze =====
            # Compute KV cache for image+text only
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            past_key_values_final = past_key_values
            prefix_pad_masks_final = prefix_pad_masks
            gaze_pixel_pred = None

        # Diffusion denoising loop for action prediction
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks_final,
                past_key_values_final,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        
        hand_pred = x_t
        return hand_pred, gaze_pixel_pred

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
