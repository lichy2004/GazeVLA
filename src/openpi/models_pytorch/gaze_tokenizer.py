"""Gaze Tokenizer for discretizing continuous gaze coordinates into token IDs.

This module provides a simple tokenizer that converts continuous gaze coordinates 
(normalized to [0, 1]) into discrete token IDs for use with language models in the 
Chain of Thought (CoT) framework.
"""

import torch
import torch.nn as nn


class GazeTokenizer(nn.Module):
    """Tokenizer for converting continuous gaze coordinates to discrete tokens.
    
    将连续的 gaze 坐标 [bs, 2] 离散化为 token IDs [bs, 2]，
    便于 LLM 进行自回归预测。使用 PaliGemma 词表最后的 224 个 token IDs。
    
    Args:
        num_bins: 每个维度的离散化级别（bin 数量）
        image_size: 图像尺寸，用于确保 gaze 坐标在有效范围内
        paligemma_vocab_size: PaliGemma 的原始词汇表大小（默认 257152）
    
    Examples:
        >>> tokenizer = GazeTokenizer(num_bins=224, paligemma_vocab_size=257152)
        >>> # gaze token IDs 范围: [256928, 257151] (使用 PaliGemma 词表最后的 224 个 token)
        >>> # 词表大小: 257152 (保持不变)
    """
    
    def __init__(self, num_bins: int = 224, image_size: int = 224, paligemma_vocab_size: int = 257152):
        super().__init__()
        self.num_bins = num_bins
        self.image_size = image_size
        self.paligemma_vocab_size = paligemma_vocab_size
        
        # 参考 ActionTokenizer 的实现方式
        # token IDs 使用词表最后的 num_bins 个 token
        # 例如：num_bins=224, paligemma_vocab_size=257152 时
        # gaze token IDs 范围: [256928, 257151]
        self.vocab_offset = paligemma_vocab_size - num_bins
        
        # 词表大小不变，仍为原始大小
        self.extended_vocab_size = paligemma_vocab_size
        
    def encode(self, gaze: torch.Tensor) -> torch.Tensor:
        """将连续的 gaze 坐标编码为离散的 token IDs（使用 PaliGemma 词表最后的 token）.
        
        Args:
            gaze: [bs, 2] 归一化的 gaze 坐标，范围 [0, 1]，顺序为 (h, w)
            
        Returns:
            gaze_token_ids: [bs, 2] 离散的 token IDs，
                          范围 [vocab_size - num_bins, vocab_size - 1]
                          例如：paligemma_vocab_size=257152, num_bins=224 时
                          范围为 [256928, 257151]（使用词表最后的 224 个 token）
        """
        # 确保 gaze 在有效范围内 [0, 1]
        gaze = torch.clamp(gaze, 0.0, 1.0)
        
        # 线性映射到 [0, num_bins] 并向下取整
        bin_ids = (gaze * self.num_bins).long()
        
        # 确保不会超出 bin 范围（处理 gaze=1.0 的边界情况）
        bin_ids = torch.clamp(bin_ids, 0, self.num_bins - 1)
        
        # 参考 ActionTokenizer 的实现：使用 vocab_size - discretized_action
        # 映射到词表最后的 num_bins 个 token，范围 [vocab_size - num_bins, vocab_size - 1]
        # bin_id=0 映射到 vocab_size - 1，bin_id=223 映射到 vocab_size - 224
        token_ids = self.paligemma_vocab_size - 1 - bin_ids
        
        return token_ids
    
    def decode(self, gaze_token_ids: torch.Tensor) -> torch.Tensor:
        """将离散的 token IDs 解码为连续的 gaze 坐标.
        
        Args:
            gaze_token_ids: [bs, 2] 离散的 token IDs，
                          范围 [vocab_size - num_bins, vocab_size - 1]
            
        Returns:
            gaze: [bs, 2] 恢复的连续坐标，范围 [0, 1]
        """
        # 参考 ActionTokenizer 的实现：vocab_size - action_token_ids
        # 将 token IDs 转换回 bin IDs
        bin_ids = self.paligemma_vocab_size - 1 - gaze_token_ids
        
        # 确保 bin IDs 在有效范围内 [0, num_bins - 1]
        bin_ids = torch.clamp(bin_ids, 0, self.num_bins - 1)
        
        # 将 bin ID 映射回连续空间，使用中心化策略
        # 例如：bin_id=0 对应区间 [0, 1/num_bins)，中心点为 0.5/num_bins
        gaze = (bin_ids.float() + 0.5) / self.num_bins
        
        # 确保结果在有效范围内
        gaze = torch.clamp(gaze, 0.0, 1.0)
        
        return gaze
    
    def __repr__(self):
        return (f"GazeTokenizer(num_bins={self.num_bins}, image_size={self.image_size}, "
                f"token_range=[{self.vocab_offset}, {self.paligemma_vocab_size - 1}], "
                f"vocab_size={self.extended_vocab_size})")


# 简单的单元测试函数
def test_gaze_tokenizer():
    """测试 GazeTokenizer 的编码和解码功能."""
    print("Testing GazeTokenizer...")
    
    # 测试参数：使用实际的 PaliGemma 词表大小
    num_bins = 224
    paligemma_vocab_size = 257152
    
    tokenizer = GazeTokenizer(num_bins=num_bins, image_size=224, paligemma_vocab_size=paligemma_vocab_size)
    print(f"\n{tokenizer}")
    print(f"Original PaliGemma vocab size: {paligemma_vocab_size}")
    print(f"Gaze token ID range: [{tokenizer.vocab_offset}, {paligemma_vocab_size - 1}]")
    print(f"Vocab size (unchanged): {tokenizer.extended_vocab_size}")
    
    # 测试1: 基本的编码-解码循环
    test_gaze = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.3, 0.7]])
    token_ids = tokenizer.encode(test_gaze)
    decoded_gaze = tokenizer.decode(token_ids)
    
    print(f"\n=== Test 1: Basic encode-decode ===")
    print(f"Original gaze:\n{test_gaze}")
    print(f"Token IDs:\n{token_ids}")
    print(f"Decoded gaze:\n{decoded_gaze}")
    print(f"Max reconstruction error: {(test_gaze - decoded_gaze).abs().max().item():.6f}")
    
    # 测试2: 验证 token IDs 在正确的范围内
    print(f"\n=== Test 2: Token ID range validation ===")
    boundary_gaze = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.999, 0.001]])
    boundary_tokens = tokenizer.encode(boundary_gaze)
    print(f"Boundary gaze:\n{boundary_gaze}")
    print(f"Token IDs:\n{boundary_tokens}")
    print(f"Min token ID: {boundary_tokens.min().item()} (expected >= {tokenizer.vocab_offset})")
    print(f"Max token ID: {boundary_tokens.max().item()} (expected < {paligemma_vocab_size})")
    
    # 验证 token IDs 在正确范围内 [vocab_size - num_bins, vocab_size - 1]
    assert boundary_tokens.min() >= tokenizer.vocab_offset, f"Token IDs should be >= {tokenizer.vocab_offset}"
    assert boundary_tokens.max() < tokenizer.extended_vocab_size, f"Token IDs should be < {tokenizer.extended_vocab_size}"
    assert boundary_tokens.min() >= paligemma_vocab_size - num_bins, f"Token IDs should be >= {paligemma_vocab_size - num_bins}"
    assert boundary_tokens.max() <= paligemma_vocab_size - 1, f"Token IDs should be <= {paligemma_vocab_size - 1}"
    
    # 验证 gaze tokens 确实使用词表最后的 token
    print(f"✓ Gaze tokens use last {num_bins} tokens of vocabulary: [{paligemma_vocab_size - num_bins}, {paligemma_vocab_size - 1}]")
    
    # 测试3: 批量处理
    print(f"\n=== Test 3: Batch processing ===")
    batch_gaze = torch.rand(16, 2)  # [bs=16, 2]
    batch_tokens = tokenizer.encode(batch_gaze)
    batch_decoded = tokenizer.decode(batch_tokens)
    max_error = (batch_gaze - batch_decoded).abs().max().item()
    print(f"Batch size: {batch_gaze.shape[0]}")
    print(f"Max reconstruction error: {max_error:.6f}")
    print(f"Expected max error (1/(2*num_bins)): {1.0/(2*num_bins):.6f}")
    print(f"Token ID range in batch: [{batch_tokens.min().item()}, {batch_tokens.max().item()}]")
    
    # 测试4: 验证 gaze tokens 使用词表最后的 token
    print(f"\n=== Test 4: Verify gaze tokens use last tokens of PaliGemma vocab ===")
    print(f"PaliGemma original vocab: [0, {paligemma_vocab_size - 1}]")
    print(f"Gaze tokens (last {num_bins} tokens): [{tokenizer.vocab_offset}, {paligemma_vocab_size - 1}]")
    print(f"Vocab size (unchanged): {tokenizer.extended_vocab_size}")
    print(f"Gaze tokens REPLACE the last {num_bins} tokens of PaliGemma vocabulary ✓")
    
    # 验证 token IDs 在词表最后 num_bins 个位置
    assert batch_tokens.min() >= paligemma_vocab_size - num_bins, f"Gaze token IDs should be >= {paligemma_vocab_size - num_bins}"
    assert batch_tokens.max() <= paligemma_vocab_size - 1, f"Gaze token IDs should be <= {paligemma_vocab_size - 1}"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gaze_tokenizer()

