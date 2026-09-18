#!/bin/bash
# 接力脚本: 等 SOLOv2 训练进程退出后, 自动跑其 test 全量推理(GPU1)
set -u
while pgrep -f "solov2_R50_1x_whu1024" > /dev/null 2>&1; do sleep 180; done
sleep 60
nohup bash /home/wangcheng/project/new/portable_sam2_explicit_coarse/scripts/baselines/infer_whu1024.sh solov2 1 0 > /dev/null 2>&1 &
echo "$(date '+%F %T') solov2 infer launched"
