#!/usr/bin/env python3
"""
PaliGemma权重格式转换脚本

功能：将HuggingFace格式的PaliGemma权重转换为OpenPI PI0格式
用途：加载PaliGemma预训练VLM权重，用于训练只需要VLM部分的模型

使用方法：
    python scripts/convert_paligemma_weights.py \
        --input /data/lichy/root/Models/paligemma-3b-pt-224 \
        --output /data/lichy/root/Models/openpi/openpi-assets/checkpoints/paligemma-3b-pt-224 \
        --reference /data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_base_pytorch
"""

import argparse
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def setup_logging():
    """配置日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def load_paligemma_shards(paligemma_dir: Path) -> dict:
    logging.info(f"从 {paligemma_dir} 加载PaliGemma权重...")
    
    paligemma_state_dict = {}
    shard_files = list(paligemma_dir.glob("model-*-of-*.safetensors"))
    
    if not shard_files:
        single_file = paligemma_dir / "model.safetensors"
        if single_file.exists():
            logging.info(f"  发现单个权重文件: {single_file.name}")
            paligemma_state_dict = load_file(str(single_file), device="cpu")
        else:
            raise FileNotFoundError(f"在 {paligemma_dir} 中未找到权重文件")
    else:
        logging.info(f"  发现 {len(shard_files)} 个分片文件")
        for shard_file in sorted(shard_files):
            logging.info(f"  加载分片: {shard_file.name}")
            shard_dict = load_file(str(shard_file), device="cpu")
            paligemma_state_dict.update(shard_dict)
    
    logging.info(f"  总共加载 {len(paligemma_state_dict)} 个权重参数")
    return paligemma_state_dict


def convert_paligemma_key_to_pi0(paligemma_key: str) -> str:
    if paligemma_key.startswith("language_model.model."):
        stripped = paligemma_key.replace("language_model.model.", "", 1)
        return f"paligemma_with_expert.paligemma.model.language_model.{stripped}"
    else:
        return f"paligemma_with_expert.paligemma.model.{paligemma_key}"


def load_pi0_reference_keys(pi0_reference_dir: Path | None) -> dict | None:
    if pi0_reference_dir is None:
        return None
    
    logging.info(f"加载PI0参考权重（用于形状验证）: {pi0_reference_dir}")
    
    pi0_model_path = pi0_reference_dir / "model.safetensors"
    if not pi0_model_path.exists():
        logging.warning(f"  未找到PI0参考文件: {pi0_model_path}")
        return None
    
    pi0_keys_shapes = {}
    with safe_open(str(pi0_model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_slice = f.get_slice(key)
            pi0_keys_shapes[key] = tensor_slice.get_shape()
    
    logging.info(f"  加载了 {len(pi0_keys_shapes)} 个PI0参考键名")
    return pi0_keys_shapes


def convert_weights(
    paligemma_state_dict: dict,
    pi0_reference_keys: dict | None = None,
    dtype: torch.dtype = torch.bfloat16
) -> dict:
    logging.info("开始转换权重...")
    
    pi0_state_dict = {}
    stats = {
        'converted': 0,
        'shape_mismatch': 0,
        'skipped': 0
    }
    skipped_keys = []
    
    for paligemma_key, value in paligemma_state_dict.items():
        pi0_key = convert_paligemma_key_to_pi0(paligemma_key)
        
        if pi0_reference_keys is not None:
            if pi0_key in pi0_reference_keys:
                ref_shape = pi0_reference_keys[pi0_key]
                if tuple(value.shape) != tuple(ref_shape):
                    logging.warning(
                        f"  形状不匹配，跳过 {paligemma_key}\n"
                        f"    期望形状: {ref_shape}, 实际形状: {value.shape}"
                    )
                    stats['shape_mismatch'] += 1
                    skipped_keys.append(paligemma_key)
                    continue
            else:
                logging.debug(f"  PI0参考中不存在该键: {pi0_key}")
                stats['skipped'] += 1
                continue
        
        pi0_state_dict[pi0_key] = value.to(dtype)
        stats['converted'] += 1
    
    logging.info(f"转换完成:")
    logging.info(f"  成功转换: {stats['converted']} 个参数")
    logging.info(f"  形状不匹配: {stats['shape_mismatch']} 个参数")
    logging.info(f"  跳过: {stats['skipped']} 个参数")
    
    if skipped_keys:
        display_keys = skipped_keys[:10] if len(skipped_keys) > 10 else skipped_keys
        logging.info(f"  跳过的键: {display_keys}" + ("..." if len(skipped_keys) > 10 else ""))
    
    return pi0_state_dict


def save_converted_weights(output_dir: Path, state_dict: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    
    logging.info(f"保存转换后的权重到: {output_path}")
    save_file(state_dict, str(output_path))
    logging.info(f"  保存了 {len(state_dict)} 个参数")
    logging.info(f"  文件大小: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="将PaliGemma权重转换为PI0格式")
    parser.add_argument(
        "--input",
        type=Path,
        default="/data/lichy/root/Models/paligemma-3b-pt-224",
        help="PaliGemma模型目录（包含分片文件或model.safetensors）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/data/lichy/root/Models/openpi/openpi-assets/checkpoints/paligemma-3b-pt-224",
        help="输出目录（将创建model.safetensors文件）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="输出权重的数据类型（默认: bfloat16）"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default="/data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_base_pytorch",
        help="PI0参考模型目录（可选，用于形状验证）"
    )
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 数据类型映射
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]
    
    logging.info("=" * 80)
    logging.info("PaliGemma权重格式转换工具")
    logging.info("=" * 80)
    logging.info(f"输入目录: {args.input}")
    logging.info(f"输出目录: {args.output}")
    logging.info(f"参考目录: {args.reference if args.reference else '未提供'}")
    logging.info(f"数据类型: {args.dtype}")
    logging.info("=" * 80)
    
    # 1. 加载PaliGemma权重
    paligemma_state_dict = load_paligemma_shards(args.input)
    
    # 2. 加载PI0参考（可选）
    pi0_reference_keys = load_pi0_reference_keys(args.reference)
    
    # 3. 转换权重
    pi0_state_dict = convert_weights(paligemma_state_dict, pi0_reference_keys, dtype)
    
    # 4. 保存转换后的权重
    save_converted_weights(args.output, pi0_state_dict)
    
    logging.info("=" * 80)
    logging.info("转换完成！")
    logging.info(f"可以在训练配置中使用: pytorch_weight_path='{args.output}'")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

