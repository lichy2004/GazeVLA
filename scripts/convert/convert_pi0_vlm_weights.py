#!/usr/bin/env python3
"""
PI0 VLM权重提取脚本

功能：从完整的PI0模型权重中提取VLM (PaliGemma)部分
用途：提取预训练的VLM权重，其他部分（action expert和投影层）随机初始化

使用方法：
    python scripts/convert_pi0_vlm_weights.py \
        --input /data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_base_pytorch \
        --output /data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_vlm
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


def load_pi0_weights(pi0_dir: Path) -> dict:
    logging.info(f"从 {pi0_dir} 加载PI0权重...")
    
    model_path = pi0_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"在 {pi0_dir} 中未找到 model.safetensors 文件")
    
    logging.info(f"  加载权重文件: {model_path.name}")
    pi0_state_dict = load_file(str(model_path), device="cpu")
    
    logging.info(f"  总共加载 {len(pi0_state_dict)} 个权重参数")
    return pi0_state_dict


def filter_vlm_weights(pi0_state_dict: dict) -> dict:
    logging.info("开始过滤VLM权重...")
    
    # VLM权重的前缀
    vlm_prefix = "paligemma_with_expert.paligemma."
    
    vlm_state_dict = {}
    discarded_keys = []
    
    # 遍历所有权重，只保留VLM相关的
    for key, value in pi0_state_dict.items():
        if key.startswith(vlm_prefix):
            vlm_state_dict[key] = value
        else:
            discarded_keys.append(key)
    
    logging.info(f"过滤完成:")
    logging.info(f"  保留VLM权重: {len(vlm_state_dict)} 个参数")
    logging.info(f"  丢弃其他权重: {len(discarded_keys)} 个参数")
    
    # 显示一些被丢弃的键名示例（用于验证）
    if discarded_keys:
        display_keys = discarded_keys[:10] if len(discarded_keys) > 10 else discarded_keys
        logging.info(f"  丢弃的键示例: {display_keys}" + ("..." if len(discarded_keys) > 10 else ""))
    
    return vlm_state_dict


def save_vlm_weights(output_dir: Path, state_dict: dict):
    """
    保存VLM权重到指定目录
    
    Args:
        output_dir: 输出目录路径
        state_dict: VLM权重字典
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    
    logging.info(f"保存VLM权重到: {output_path}")
    save_file(state_dict, str(output_path))
    logging.info(f"  保存了 {len(state_dict)} 个参数")
    logging.info(f"  文件大小: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    """主函数：解析参数并执行VLM权重提取"""
    parser = argparse.ArgumentParser(description="从PI0权重中提取VLM部分")
    parser.add_argument(
        "--input",
        type=Path,
        default="/data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_base_pytorch",
        help="PI0模型检查点目录（包含model.safetensors文件）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/data/lichy/root/Models/openpi/openpi-assets/checkpoints/pi0_vlm",
        help="输出目录（将创建model.safetensors文件）"
    )
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("PI0 VLM权重提取工具")
    logging.info("=" * 80)
    logging.info(f"输入目录: {args.input}")
    logging.info(f"输出目录: {args.output}")
    logging.info("=" * 80)
    
    # 1. 加载PI0权重
    pi0_state_dict = load_pi0_weights(args.input)
    
    # 2. 过滤VLM权重
    vlm_state_dict = filter_vlm_weights(pi0_state_dict)
    
    # 3. 保存VLM权重
    save_vlm_weights(args.output, vlm_state_dict)
    
    logging.info("=" * 80)
    logging.info("VLM权重提取完成！")
    logging.info(f"可以在训练配置中使用: pytorch_weight_path='{args.output}'")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

