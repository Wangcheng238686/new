#!/bin/bash
# 接力脚本: 等 GPU2 上的 SCNet v3 训练进程退出后, 自动挂 Mask2Former 300ep 延长版
set -u
while pgrep -f "scnet_r50_fpn_1x_whu1024" > /dev/null 2>&1; do sleep 300; done
sleep 60  # 等 GPU 显存完全释放
nohup bash /home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines/run_mmdet_whu1024.sh mask2former train 2 > /dev/null 2>&1 &
echo "$(date '+%F %T') mask2former 300ep launched after scnet"
