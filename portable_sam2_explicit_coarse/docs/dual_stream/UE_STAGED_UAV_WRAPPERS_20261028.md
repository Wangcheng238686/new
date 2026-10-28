# 三阶段实验一键 wrapper 与单变量审查

日期：2026-10-28。对应 [已确认的收紧路线](UE_COMPACT_EXPERIMENT_ROADMAP_20261028.md)。本次只准备脚本、CPU检查和源码冻结，不启动正式训练。

## 1. 唯一入口

`/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/scripts/experiments/ue_staged_uav.sh`

默认输出根：`/data/wangcheng/checkpoint/ue_staged_uav_full100_20261028`。所有命令可显式传 `--root`；同一次研究必须保持相同 root。`UE_EXPERIMENT_PYTHON` 可指定同环境 Python，默认 `/data/wangcheng/envs/cvt2/bin/python`。GPU始终0/1/2/3四卡，无单卡回退。

使用时将下列 `WRAPPER` 替换为上述绝对路径，或在 shell 中设置：

```bash
WRAPPER=/home/wangcheng2021/project/new-uav-dual/portable_sam2_explicit_coarse/scripts/experiments/ue_staged_uav.sh
"$WRAPPER" audit
"$WRAPPER" start --stage structure
```

`start` 一次完成 prepare→当前阶段cache准备/核验→后台queue。全量视角缓存构建在CPU进行，可能耗时；随后GPU owner按四卡共同余量等待/共租。不跨阶段自动推进。`plan/audit` 只解析配置与差异，不创建训练或CUDA任务。

## 2. 三阶段操作

### 第一阶段：结构

```bash
"$WRAPPER" start --stage structure
"$WRAPPER" status
```

S/U/T/C全部先通过各自工程门及共同初始化核对，再启动100轮训练。**结构队列只做gate和train，不自动产出test。** 四组训练结束后，根据val与工程证据明确选择保留结构，再放行test：

```bash
# U仅为语法示例，不是本次替用户选定结构；可以选U、T或C。
"$WRAPPER" lock --base-arm U --reason "填写验证集与工程证据依据"
"$WRAPPER" eval --stage structure
"$WRAPPER" analyze --stage structure
```

lock写入不可变selection_lock，绑定四组训练收据、所选val-best、检查点SHA及代码/协议，不读取test排名。未锁定时，controller和直接runner均拒绝结构test。锁定后不因test结果更换结构；确需新研究必须独立root并披露探索。

### 第二阶段：模态

```bash
"$WRAPPER" start --stage modality
"$WRAPPER" analyze --stage modality
```

仅新增 `mod_constant` 和 `mod_no_geometry` 两组；完整RGB-D为锁定的U/T/C基座，直接复用第一阶段结果。两组全部训练完才开始本阶段test。不会生成“双关”、平面或逐入口交叉实验。

### 第三阶段：视角

```bash
"$WRAPPER" start --stage views
"$WRAPPER" analyze --stage views
```

仅新增 `views_1=[0]`、`views_3=[0,144,288]`，五视角完整模型复用锁定基座。正式子集缓存必须覆盖全3060图且从所选raw深度重建。只有研究者确认已有相对S净收益、决定推进采集预算研究后才执行该阶段；脚本不自动根据test判断或发射。

`analyze` 为显式CPU步骤，不由queue自动运行。统一Mask/BBox/BoundaryAP2%/1%，同459图500次seed20261021配对bootstrap；结构另给同抽样的C−U−T+S。输出 `root/analysis/STAGE/summary.json`，拒绝覆盖完整结果。该分析不参与loss/门控/选权。

## 3. 细分与恢复操作

```bash
"$WRAPPER" prepare
"$WRAPPER" cache --stage structure
"$WRAPPER" queue --stage structure
"$WRAPPER" gate --stage structure --recipe S
"$WRAPPER" train --stage structure --recipe S
"$WRAPPER" resume --stage structure --recipe S
"$WRAPPER" recover --stage structure
```

单臂train也要求本阶段全部工程门通过；不能只gate S后直接购买全量预算。resume只接受本root/同recipe的last.pth、epoch边界、相同源码/输入/订单/协议；恢复optimizer、AMP、Python/NumPy/Torch/CUDA/DataLoader generator五流。recover检查旧launcher及四rank真实PID/start_ticks，不接管仍活跃任务。结构test队列恢复使用 `recover --stage structure --evaluation-only`。

prepare建立独立git detached工作树和100轮共同样本订单，后续源码变化不会进入已准备研究。拒绝覆盖已有不一致root。当前旧ABC任务在另一冻结工作树，完全不重启或改动。

旧B/C可能对应U/C，但新wrapper不会自动改写历史合同、导入已训练权重或伪造新marker。历史复用必须另核数值/数据/初始化/预算等价；未核定之前不能直接把旧产物放入本root。启动前应先决定可复用项，避免重复训练。

## 4. 实际单变量审查

| 比较 | 唯一声明因子 | 固定项／解释边界 |
|---|---|---|
| S→U、T→C | 上游U系统包 | 包含早期写回、交互、实例loss和补框；不是单个融合算子 |
| S→T、U→C | UBR的UAV证据入口 | UBR结构、原主任务不变；不是删除整个尾部头 |
| 基座→mod_constant | 全路径RGB内容 | 相同编码器、LoRA、深度对应、形状内容与损失；源RGB统一换固定均值 |
| 基座→mod_no_geometry | 全路径显式几何内容 | geometry5[:4]和candidate7[2,3,6]归零；保留support、index/anchor、offset/yaw |
| 基座→views_1/3 | 所选相机子集 | 同容量、候选上限和协议；RGB/深度/派生候选与几何同步变化，缓存是该因子的派生产物 |

S→C同时改变U和T，只作完整系统比较。S/T没有UAV实例loss、U/C有，这是U系统包的预期差异；不能声称四组所有loss完全相同。模态/视角阶段才要求保留结构及全部既有loss/权重一致。

几何关闭不是无深度，仍有实测深度建立的对应和有效性；常量RGB不是纯深度模型，可能消费卫星条件信息。视角只比较预定子集，不声称独立于拍摄方向的普遍数量因果效应；源dropout随机流可能因编码视角数变化而不同，不能冒称全CUDA轨迹配对相同。

结构卫星/UBR初始SHA须跨四组相同，U/T/C共享RGB初始SHA相同；后续对照完整模型初始SHA须与所选基座相同。recipe schema拒绝隐藏配置字段或同时改变两个内容开关。训练均从同通用初始化开始，禁止以工程门或已训练分支热启动。

## 5. 固定训练与资源合同

沿用已批准的全100轮数值合同：26700成功更新，四卡各batch1×累积2，全局8；AdamW lr5e-4、wd.05、warmup100、cosine至.1、clip.5、seed44、AMP训练/FP32评价、无EMA、实例上限250。coarse64、native128、原mask监督512、增广及原任务loss不变。每轮完整459 val，按val MaskAP严格更大选best（相等取早），test完整459。

新协议只修正阶段和因子描述，不重写上述数字。epoch订单显式固定44+epoch，恢复后核顺序。正式主模型任务不增加UBR/边界/模态新loss。

未知recipe先预留15872MiB/card并另留共同1024；工程门实测后预约max(peak_reserved+1024,resident+256)，所有卡满足才准入，启动中尚未兑现的预约会预扣。允许多个完整四卡任务共租，不抢占外部任务；不足时后台等待。allocator cap默认0，不随臂改变训练算法。

每recipe独立gate、CP、日志、test和锁；所有阶段源码、配置、recipe、订单、缓存SHA严格绑定。S的缓存收据只核卫星数据身份，不读UAV NPZ。直接torchrun也必须通过cache/selection检查。

## 6. 验收状态

- CPU测试覆盖配置因子、全路径模态控制、S/T路由、队列/锁定/共租预扣、检查点/缓存篡改拒绝、初始化家族、离线配对分析及既有UBR/子集回归；88项测试通过；CPU prepare亦已通过，详见下方记录。
- shell语法和实际audit命令检查；独立代理审查发现的“锁定前暴露test”“直接runner绕过缓存”“默认root不一致”“S依赖无关UAV缓存”已修。
- 新四卡真实gate/资源/下一步训练轨迹未运行，本次不宣称GPU验收、AP增益或正式续训已通过。GPU仍由现有ABC共租占用。
- 任何后续stage开始前都先通过本stage真实工程门；门只验证接线/资源，不能代替内容净收益证据。


### 最终CPU准备验收

数值/控制源码提交 `a95a444`（主实现 `6e985ad`），正式待启动根目录已准备：`/data/wangcheng/checkpoint/ue_staged_uav_full100_20261028`。detached runtime_source冻结536个源码/配置文件；100轮订单及manifest共101份文件验证通过，实际本地导入闭包匹配，S缺cache验收的直接runner调用正确拒绝。88项CPU测试通过，CUDA未初始化，正式训练未发射。

机器收据：`/data/wangcheng/checkpoint/ue_staged_uav_full100_20261028/CPU_PREPARE_AUDIT.json`。当前cache_receipts为空、base_lock为空是预期准备状态；后续start先做当前阶段缓存验收与四卡gate。

第一次CPU准备因PyTorch动态命名空间伪`_ops.py`被误判为真实未冻结源码而失败；源码文件实际不存在。已增加is_file筛选与回归测试，从新提交重新准备。旧失败产物完整保留在`/data/wangcheng/checkpoint/ue_staged_uav_full100_20261028_prepare_attempt1`，未修改旧源码字节；该目录仅审计，不可用于启动。本次新冻结源未改动当前ABC工作树。
