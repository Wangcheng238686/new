# 恢复命令（当前仅静态就绪，不代表准入）

当前GPU/内存/I/O有UE驻留负载，主代理要求不启动timing。正式协议与DRAFT同字节，精确SHA见source_ledger.json。source一旦修改，必须审核并重新登记，不复用旧准入。

1. 先由主代理核环境及独立审查，恢复唯一提取六图smoke：

```bash
CUDA_VISIBLE_DEVICES=<核准GPU> /data/wangcheng/envs/cvt2/bin/python /tmp/ue_projection_scale_probe/extract_projection_scale.py --smoke
```

若共享GPU必须沿主代理launch_smoke.py显存cap流程，不擅自移除allocator cap。此前未完成smoke不作收益结果。

2. 六图cache独立验收PASS、audit明确准入后，唯一train-only32step计时（不读validation targets）：

```bash
CUDA_VISIBLE_DEVICES=<核准GPU> /data/wangcheng/envs/cvt2/bin/python /tmp/ue_projection_scale_fit.py --features /tmp/ue_projection_scale_probe/smoke/run_manifest.json --timing-smoke --device cuda:0
```

3. root另准入完整281冻结提取后：

```bash
CUDA_VISIBLE_DEVICES=<核准GPU> /data/wangcheng/envs/cvt2/bin/python /tmp/ue_projection_scale_probe/extract_projection_scale.py
```

4. 完整cache/资源/身份独立验收PASS、root另准入1000step12head fit后：

```bash
CUDA_VISIBLE_DEVICES=<核准GPU> /data/wangcheng/envs/cvt2/bin/python /tmp/ue_projection_scale_fit.py --device cuda:0
```

正式拟合始终fresh本目录权重，不从旧stage权重热启；同arch init/样本序列/seed1831、AdamW1e−3 wd1e−4、1000steps last。不据收益改步数/宽度/seed。

5. 已完成固定fit后CPU summary：

```bash
OPENBLAS_NUM_THREADS=2 /home/wangcheng2021/anaconda3/bin/python /tmp/ue_projection_scale_summary.py
```

四主比较98.75%CI+0.001门保持；Fine−coarse95%仅描述。F双支持门均过且Fine−coarse下界>0才允许称该matched条件细尺度更好，不称生产分割/AP或不可逆信息损失。

## 实际状态更新

唯一6图smoke已自然exit0，6图240ROI独立CPU验收PASS；不重跑已有smoke。上文步骤1为未来若source/协议发生改变才需重新准入的命令，不代表当前缺少smoke。当前尚缺train-only timing、完整281提取/验收、固定读出及最终结果审核。资源IO full约72%/swap35GiB时未准full/fit。恢复前重新核环境与源SHA；若仍共享UE，保留allocator20%约4.9GiB硬cap并把初始化纳入外部wall900秒预算；不终止UE，不无条件从命令直接跳级。
