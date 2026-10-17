# EXP-0018/0019 全量启动审计（2026-10-17）

审计对象：`aff2ab6`（tags EXP-0018/EXP-0019）；独立子代理审计 2026-10-17
完成（Explore，只读）。**VERDICT: PASS**，无阻断问题。

要点（全文见子代理会话输出 sess_463660c0/agent_d19d87f2）：

1. wrapper 全部守卫与旗标核验：B0_EXP_ID→seed/GPU 映射、clean-tree、
   数据双哈希、身份 `ue-whu-a3sgm-512-warm-r1g-v1`、torchrun 旗标与筛查
   wrapper 逐串一致（唯一 delta = 预注册的世界 4→2 / 累积 2→4 / 轮数
   30→100）；init/freeze 旗标与筛查 fresh 分支逐字节一致。
2. 批量/步数数学在代码中核实：DDP2 694/2=347 minibatch/epoch、87 优化步
   （86 全组+1 部分组按 6 样本正确归一化）、global batch 8、单余弦
   T_max=8600 warmup 含内；与 DDP4 锚点（696 padding、174/rank、87 步）
   在预注册可比维度逐值一致。
3. DDP2-vs-DDP4 偏差均在预注册"shard 组成"包络内：val rank0-only 不变
   （非分片、确定性 100 场，与世界大小无关）、无 SyncBN、set_epoch 正常。
4. test 封存核实：加载路径只构造 train/val，create_test_loader 未被引用。
5. 无 resume、无 unlock；SAVE_LAST_MODEL 产出预注册选择工件
   last_model_epoch100.pth；冻结语义为 cdc9e822 修正版（trainable =
   前缀 ∩ 构建时可训）。
6. 登记表两行与 wrapper 一致；RUN_TAG seed/EXP-ID 双不相交。

操作性 caveat（采纳）：wrapper 的 CUDA_VISIBLE_DEVICES 默认映射可被外层
shell 覆盖——两 wrapper 启动时显式指定（0,1 / 2,3），已在启动命令执行。
