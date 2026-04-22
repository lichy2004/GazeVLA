#!/bin/bash

# Robotwin2 数据集 Norm Stats 批量计算脚本
# 为所有 easy, hard, multi (all) 难度的任务计算归一化统计信息

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Robotwin2 Norm Stats 批量计算${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Easy 难度任务
easy_tasks=(
    "adjust_bottle_easy"
    "beat_block_hammer_easy"
    "blocks_ranking_rgb_easy"
    "blocks_ranking_size_easy"
    "click_alarmclock_easy"
    "click_bell_easy"
    "dump_bin_bigbin_easy"
    "grab_roller_easy"
    "handover_block_easy"
    "handover_mic_easy"
    "hanging_mug_easy"
    "lift_pot_easy"
    "move_can_pot_easy"
    "move_pillbottle_pad_easy"
    "move_playingcard_away_easy"
    "move_stapler_pad_easy"
    "open_laptop_easy"
    "open_microwave_easy"
    "pick_diverse_bottles_easy"
    "pick_dual_bottles_easy"
    "place_a2b_left_easy"
    "place_a2b_right_easy"
    "place_bread_basket_easy"
    "place_bread_skillet_easy"
    "place_burger_fries_easy"
    "place_can_basket_easy"
    "place_cans_plasticbox_easy"
    "place_container_plate_easy"
    "place_dual_shoes_easy"
    "place_empty_cup_easy"
    "place_fan_easy"
    "place_mouse_pad_easy"
    "place_object_basket_easy"
    "place_object_scale_easy"
    "place_object_stand_easy"
    "place_phone_stand_easy"
    "place_shoe_easy"
    "press_stapler_easy"
    "put_bottles_dustbin_easy"
    "put_object_cabinet_easy"
    "rotate_qrcode_easy"
    "scan_object_easy"
    "shake_bottle_easy"
    "shake_bottle_horizontally_easy"
    "stack_blocks_three_easy"
    "stack_blocks_two_easy"
    "stack_bowls_three_easy"
    "stack_bowls_two_easy"
    "stamp_seal_easy"
    "turn_switch_easy"
)

# Multi 难度任务（所有 _all 任务）
multi_tasks=(
    "all_all"
)

# 合并所有任务
all_tasks=("${easy_tasks[@]}" "${hard_tasks[@]}" "${multi_tasks[@]}")

# 统计信息
total_tasks=${#all_tasks[@]}
current_task=0
failed_tasks=()

echo -e "${BLUE}总任务数: ${total_tasks}${NC}"
echo ""

# 遍历所有任务并计算 norm stats
for task in "${all_tasks[@]}"; do
    current_task=$((current_task + 1))
    config_name="pi0_${task}"
    
    echo -e "${BLUE}[${current_task}/${total_tasks}]${NC} 正在处理: ${GREEN}${config_name}${NC}"
    
    # 执行 compute_norm_stats 命令
    if python scripts/compute_norm_stats.py --config-name "${config_name}" --max_frames 100000; then
        echo -e "${GREEN}✓ 完成: ${config_name}${NC}"
    else
        echo -e "${RED}✗ 失败: ${config_name}${NC}"
        failed_tasks+=("${config_name}")
    fi
    
    echo ""
done

# 输出总结
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}计算完成！${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "总任务数: ${total_tasks}"
echo -e "${GREEN}成功: $((total_tasks - ${#failed_tasks[@]}))${NC}"
echo -e "${RED}失败: ${#failed_tasks[@]}${NC}"

# 如果有失败的任务，列出它们
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}失败的任务:${NC}"
    for task in "${failed_tasks[@]}"; do
        echo -e "  - ${task}"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}所有任务已成功完成！${NC}"