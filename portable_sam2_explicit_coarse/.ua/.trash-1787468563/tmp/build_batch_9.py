#!/usr/bin/env python3
"""Builder for batch-9 graph fragments (deterministic, self-validating)."""
import json
import math
import os
from collections import OrderedDict

OUT_DIR = "/home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/intermediate"

# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------
def F(path, name, summary, tags, complexity, languageNotes=None):
    n = OrderedDict(id=f"file:{path}", type="file", name=name, filePath=path,
                    summary=summary, tags=tags, complexity=complexity)
    if languageNotes:
        n["languageNotes"] = languageNotes
    return n

def FN(path, fname, line_range, summary, tags, complexity, languageNotes=None):
    n = OrderedDict(id=f"function:{path}:{fname}", type="function", name=fname,
                    filePath=path, lineRange=line_range, summary=summary,
                    tags=tags, complexity=complexity)
    if languageNotes:
        n["languageNotes"] = languageNotes
    return n

def CL(path, cname, line_range, summary, tags, complexity, languageNotes=None):
    n = OrderedDict(id=f"class:{path}:{cname}", type="class", name=cname,
                    filePath=path, lineRange=line_range, summary=summary,
                    tags=tags, complexity=complexity)
    if languageNotes:
        n["languageNotes"] = languageNotes
    return n

# ============================ PART 1 NODES =================================
P1 = "configs/_sam2_registry.py"
P1S = "configs/rsprompter_anchor_satS_v11_sam2_large_full.py"
INF = "inference/infer_from_checkpoint.py"
ARC = "rsprompter/architecture_contract.py"

part1_nodes = [
    F("configs/_sam2_registry.py", "_sam2_registry.py",
      "SAM2 模型规格的唯一配置来源（single source of truth）：统一解析 model_size、checkpoint 路径、结构 YAML 与 neck in_channels 的映射，修复此前多处硬编码导致的 baseplus/large 名实不符问题。",
      ["configuration", "sam2", "registry", "model-resolution"], "moderate"),
    CL("configs/_sam2_registry.py", "SAM2ModelConfig", [37, 47],
       "SAM2 模型规格 dataclass（单一真相源）：持有 model_size、checkpoint、yaml 与 neck_in_channels 四个字段，并提供 model_size_is_large 属性。",
       ["data-model", "sam2", "config"], "simple",
       "frozen dataclass 保证规格对象不可变。"),
    FN("configs/_sam2_registry.py", "build_sam2_cfg", [80, 104],
       "按 env SAM2_MODEL_SIZE / SAM2_CKPT 优先、默认 large 兜底的顺序构建 SAM2ModelConfig 并校验 checkpoint 存在性，是各训练 config 获取 SAM2 规格的入口。",
       ["factory", "sam2", "config", "env-driven"], "moderate"),
    FN("configs/_sam2_registry.py", "sha256_of_file", [107, 118],
       "分块读取文件并计算 SHA-256 摘要，用于预训练 checkpoint 的身份指纹。",
       ["utility", "hash", "checkpoint"], "simple"),

    F("configs/environment.sh", "environment.sh",
      "机器级默认环境变量脚本：集中定义 Python 解释器、SAM2 repo/checkpoint、WHU 数据根目录、checkpoint/log/tmp 输出根以及 CUDA 设备与 DDP 进程数等执行默认值，外层 shell 已导出的变量优先。",
      ["configuration", "environment", "shell", "defaults"], "simple"),

    F("configs/rsprompter_anchor_satS_v11_sam2_large_full.py",
      "rsprompter_anchor_satS_v11_sam2_large_full.py",
      "RSPrompter + SAM2-Hiera 完整检测器根 config：定义 RSPrompterAnchor 两阶段模型（RSSAM2VisionEncoder LoRA backbone、NECK_TYPE 可切换 PAFPN/Aggregator、RPN + RSPrompterAnchorRoIPromptHead + RSPrompterAnchorMaskHeadSAM2）、512 分辨率 SatDataset 数据管道与 AdamW + warmup/cosine 训练策略，是所有 WHU1024 config 的继承基座。",
      ["configuration", "mmdet", "model-definition", "base-config", "sam2"], "complex",
      "mmengine Config.fromfile 以 eval 执行本文件（无 __file__），因此通过 cwd 候选路径用 importlib 动态加载 _sam2_registry。"),
    FN("configs/rsprompter_anchor_satS_v11_sam2_large_full.py", "_load_sam2_registry", [24, 45],
       "按 cwd 候选路径列表定位 configs/_sam2_registry.py 并用 importlib 加载，规避 mmengine eval 执行环境中缺少 __file__ 的问题。",
       ["utility", "config-loading", "importlib"], "moderate"),

    F("configs/whu1024_baseplus_clean.py", "whu1024_baseplus_clean.py",
      "干净的 WHU-1024 / SAM2 Base+ 迁移基线 config：继承 satS_v11 根 config 并剔除 IIMR/SABL/UAV 多视角分支，改用 1024 分辨率 WHUCocoSingleClass 数据集，支持 embedding stride 16/32 与 roi_local/full_image final-mask 坐标模式切换。",
      ["configuration", "baseline", "whu", "dataset", "mmdet"], "moderate"),

    F("configs/whu1024_baseplus_explicit_coarse.py", "whu1024_baseplus_explicit_coarse.py",
      "当前实验 config：在干净 WHU-1024 Base+ 基线上启用显式粗掩码提示路线（explicit coarse-mask prompt），通过环境变量矩阵（EXPLICIT_PROMPT_MODE、SHAPE_DENSE_TRANSFORM、SHAPE_LOSS_SCHEDULE_MODE、FINAL_MASK_LOSS_MODE 等）参数化 mask_head 的 shape prior 注入、shape point 挖掘、dense prompt 变换、coarse mask 监督调度、P2 boundary refiner、ROI-SAM 与 point warmup 等开关。",
      ["configuration", "experiment", "explicit-coarse", "prompt-learning", "ablation"], "complex",
      "env 驱动的 config 参数化模式：同一份 config 借助环境变量矩阵覆盖整族消融实验，读取处均带取值校验快速失败。"),

    F("configs/whu1024_baseplus_explicit_coarse_densefix.py",
      "whu1024_baseplus_explicit_coarse_densefix.py",
      "explicit_coarse config 的 C4 densefix 消融覆盖层：把 shape prior 的缩放门改为 fixed 模式（初值 0.5），并可选择开启 PromptEncoder 的 train_mask_downscaling。",
      ["configuration", "ablation", "dense-prompt", "override"], "simple"),

    F("inference/__init__.py", "__init__.py",
      "inference 包的 __init__，仅含 docstring，声明该包承载 explicit-coarse 主线的 checkpoint 驱动推理。",
      ["package-init", "inference", "documentation"], "simple"),

    F("inference/infer_from_checkpoint.py", "infer_from_checkpoint.py",
      "自描述 checkpoint 推理入口：从训练 checkpoint 内嵌的 config_snapshot 与架构契约重建模型（旧 checkpoint 可回退 --config），恢复 SAM2 repo/checkpoint 环境并载入权重，随后在 WHU COCO split 上推理，输出 COCO RLE 预测与 JSON 诊断报告。",
      ["entry-point", "inference", "checkpoint", "coco", "serialization"], "complex"),
    FN("inference/infer_from_checkpoint.py", "parse_args", [31, 79],
       "解析推理 CLI：checkpoint 路径、split 选择（validation/test/custom）、数据根与标注覆盖、设备与输出目录等。",
       ["cli", "argument-parsing", "inference"], "moderate"),
    FN("inference/infer_from_checkpoint.py", "_restore_architecture_environment", [134, 151],
       "从 checkpoint 快照恢复 SAM2_REPO/SAM2_CKPT 等架构运行环境变量，保证跨机器推理时模型构建输入一致。",
       ["environment", "restore", "checkpoint"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_restore_embedded_architecture_environment", [154, 176],
       "从 model_config 内嵌的 runtime_config 恢复 SAM2 环境变量，是快照路径缺失时的兜底恢复路径。",
       ["environment", "restore", "fallback"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_resolve_sam2_repo", [179, 195],
       "定位 SAM2 仓库路径：优先 args 与快照设置，逐级回退到仓库根/环境变量，找不到时给出可诊断的错误。",
       ["path-resolution", "sam2", "environment"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_targeted_sam2_checkpoint_override", [198, 217],
       "在重建模型前把快照中的 SAM2 checkpoint 路径定向覆盖到当前机器路径，避免旧 checkpoint 内的绝对路径失效。",
       ["checkpoint", "override", "sam2"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_resolve_model_config", [220, 333],
       "核心模型配置解析：优先使用 checkpoint 内嵌 model_config，旧 checkpoint 回退 --config 文件，校验架构契约一致性（含 SAM2 权重 sha256）后返回可直接 MODELS.build 的配置。",
       ["config-resolution", "architecture-contract", "model-building"], "complex"),
    FN("inference/infer_from_checkpoint.py", "_register_and_build", [336, 353],
       "导入 rsprompter 触发 mmdet 注册表注册，再按 model_config 用 MODELS.build 重建检测器。",
       ["model-building", "registry", "mmdet"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_select_state_dict", [356, 369],
       "从 checkpoint 字典中定位真实模型权重，兼容 state_dict/model 等多种键名与包装结构。",
       ["checkpoint", "state-dict", "utility"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_load_model_state", [372, 384],
       "以 strict/非 strict 两种模式把权重载入重建的模型并报告缺失/意外键。",
       ["checkpoint", "state-dict", "loading"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_resolve_dataset_contract", [387, 442],
       "依据 checkpoint 的 data_config 解析推理数据集契约：split、标注文件、图像子目录与 image size，允许 CLI 覆盖。",
       ["dataset", "config-resolution", "inference"], "moderate"),
    FN("inference/infer_from_checkpoint.py", "_build_loader", [445, 470],
       "按数据集契约构造推理 DataLoader（batch size、num_workers、max_batches）。",
       ["dataloader", "inference", "coco"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_build_data_samples", [473, 506],
       "把 batch 图像、GT 与元数据组装为 mmdet DetDataSample 并搬运到目标设备。",
       ["data-samples", "mmdet", "preprocessing"], "moderate"),
    FN("inference/infer_from_checkpoint.py", "_extract_instances_numpy", [509, 549],
       "从模型预测的 InstanceData 提取 scores/labels/bboxes/masks 并转为 numpy（含空预测与 RLE 编码处理）。",
       ["postprocessing", "predictions", "serialization"], "moderate"),
    FN("inference/infer_from_checkpoint.py", "_prediction_records", [571, 590],
       "把预测实例整理为 COCO results 记录列表，附图像 id 与元数据。",
       ["coco", "serialization", "predictions"], "simple"),
    FN("inference/infer_from_checkpoint.py", "_checkpoint_report", [593, 618],
       "生成 checkpoint 诊断报告（架构 ID、schema 版本、训练参数摘要等）并写 JSON。",
       ["reporting", "checkpoint", "diagnostics"], "simple"),
    FN("inference/infer_from_checkpoint.py", "main", [621, 802],
       "推理主流程：加载自描述 checkpoint → 恢复环境 → 重建模型并载权重 → 构建数据集 loader → 推理 → 阈值过滤 → 写出 COCO 预测与诊断报告。",
       ["entry-point", "inference", "pipeline"], "complex"),

    F("rsprompter/__init__.py", "__init__.py",
      "rsprompter 包入口：默认全量导入 models 与 models_sam2 完成 mmdet 注册表注册；设置 RSPROMPTER_LIGHT_IMPORT=1 可跳过导入以支持不实例化模型的纯张量单元测试。",
      ["entry-point", "barrel", "registry", "package-init"], "simple"),

    F("rsprompter/architecture_contract.py", "architecture_contract.py",
      "架构身份与 checkpoint 指纹的规范化工具：把 cfg.model 归一化（屏蔽本机 ckpt 路径）后哈希为 fingerprint，并按 neck、final-mask 模式、embedding stride、coarse 路线等维度生成稳定的 architecture_id，供训练与加载时的架构契约断言使用。",
      ["utility", "architecture-contract", "fingerprint", "validation"], "moderate"),
    FN("rsprompter/architecture_contract.py", "file_sha256", [16, 21],
       "分块读取文件并计算 SHA-256 摘要。",
       ["utility", "hash", "checksum"], "simple"),
    FN("rsprompter/architecture_contract.py", "architecture_fingerprint", [44, 51],
       "对归一化后的 model config 做规范 JSON 序列化并取 SHA-256，得到机器无关的架构指纹。",
       ["fingerprint", "hash", "normalization"], "simple"),
    FN("rsprompter/architecture_contract.py", "architecture_id", [54, 130],
       "由 neck 类型（pafpn/aggregator）、embedding stride、final-mask 坐标模式、coarse 路线（explicit_prompt_mode、dense transform/detach、final loss 模式、P2 refiner、ROI-SAM 等）组合出人类可读且稳定的架构 ID，未启用 coarse 时回退到已知基线 ID 表。",
       ["architecture-id", "naming", "contract", "ablation"], "complex"),
    FN("rsprompter/architecture_contract.py", "architecture_contract", [133, 138],
       "组装完整架构契约 dict（schema 版本、architecture_id、fingerprint、归一化 config），写入 checkpoint 供推理端重建模型。",
       ["contract", "checkpoint", "serialization"], "simple"),
    FN("rsprompter/architecture_contract.py", "assert_checkpoint_architecture", [141, 164],
       "加载 checkpoint 时断言其内嵌架构契约与期望一致，支持显式 allow_cross_arch 跨架构初始化豁免。",
       ["validation", "checkpoint", "contract"], "moderate"),

    F("rsprompter/ckpt_migration.py", "ckpt_migration.py",
      "仅初始化加载所用的 checkpoint 迁移工具：把旧版线性 shape gate 权重换算为现行 sigmoid raw gate（logit 变换），返回迁移后的 state_dict 与迁移清单。",
      ["checkpoint", "migration", "compatibility"], "simple"),
    FN("rsprompter/ckpt_migration.py", "migrate_legacy_shape_gate_state_dict", [8, 28],
       "将旧后缀 shape_prompt_scale 标量 clamp 到 (0,1) 后取 logit，改写为新后缀 shape_dense_alpha_raw，完成旧权重向新门控的迁移。",
       ["migration", "state-dict", "gate"], "simple"),
]

# ============================ PART 2 NODES =================================
DPU = "rsprompter/dense_prompt_utils.py"
SMOKE = "scripts/ablations/smoke_ddp_control_plane.py"
TEST = "scripts/ablations/test_dense_prompt_utils.py"
VAL = "scripts/ablations/validate_ablation_contract.py"
R1C4 = "scripts/ablations_uecoco/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh"
WAIT = "scripts/ablations_uecoco/wait_then_launch_r1_c4_rd.sh"
RESPLIT = "scripts/resplit_ue_coco.py"
M2C = "tools/mask_to_coco_whu512.py"
TRAIN = "train/train_rsprompter_fusion.py"

part2_nodes = [
    F("rsprompter/dense_prompt_utils.py", "dense_prompt_utils.py",
      "独立于 mask_head 的 dense prompt 变换与挖掘工具集：提供 ROI-local coarse logits 的 raw_logits/confidence_signed 变换、按 ROI 中心/面积生成各向同性 Gaussian 全画布 prompt、无效 ROI 回退 no-mask 嵌入与 ROI 到全图 canvas 的粘贴；处理顺序为 transform → resize → paste → outside_fill → clamp。",
      ["utility", "dense-prompt", "prompt-learning", "tensor-ops"], "complex",
      "独立模块设计使其可在不实例化整个 SAM2 模型的情况下做纯张量单元测试。"),
    FN("rsprompter/dense_prompt_utils.py", "transform_coarse_prompt", [13, 59],
       "ROI-local coarse logits 转 prompt logits：raw_logits 直通（可温度缩放、clamp、detach），confidence_signed 走 sigmoid → signed → confidence^gamma → strength 缩放。",
       ["transform", "dense-prompt", "gradient-control"], "moderate"),
    FN("rsprompter/dense_prompt_utils.py", "gaussian_prompt_from_roi_coarse", [62, 205],
       "从 coarse logits 的前景质心与 ROI 面积推导各向同性 Gaussian 的中心与 sigma，生成全画布 prompt logits；空/无效 ROI 标记 invalid 并回退 base embedding，同时返回统计量。",
       ["gaussian", "dense-prompt", "roi", "fallback"], "complex"),
    FN("rsprompter/dense_prompt_utils.py", "replace_invalid_dense_with_base", [208, 228],
       "把 valid 掩码为 False 的 ROI 的 dense embedding 逐元素替换为 no_mask base embedding，保证无效提示严格热启动。",
       ["fallback", "embedding", "tensor-ops"], "simple"),
    FN("rsprompter/dense_prompt_utils.py", "paste_roi_to_full_canvas", [231, 289],
       "把 ROI-local prompt logits 按 boxes 重采样并粘贴到 (mask_h, mask_w) 全画布，ROI 之外填充 outside_fill_logit。",
       ["paste", "roi", "canvas", "resample"], "moderate"),

    F("scripts/ablations/smoke_ddp_control_plane.py", "smoke_ddp_control_plane.py",
      "CPU/Gloo 进程组上的 DDP 控制面冒烟测试：rank 0 延迟 2 秒后广播 stop 标志，验证每个 rank 都能收到 rank-0 迟到的控制信号。",
      ["test", "ddp", "smoke-test", "control-plane"], "simple"),
    FN("scripts/ablations/smoke_ddp_control_plane.py", "main", [20, 32],
       "初始化 gloo 进程组，复现 rank-0 延迟控制场景并断言 _broadcast_stop_training 在所有 rank 生效。",
       ["test", "ddp", "gloo"], "simple"),

    F("scripts/ablations/test_dense_prompt_utils.py", "test_dense_prompt_utils.py",
      "dense_prompt_utils 的纯张量 unittest 套件（RSPROMPTER_LIGHT_IMPORT=1 免加载模型）：覆盖 detach 梯度控制、Gaussian 中心/面积到全画布映射、各向异性 box 在图像空间的各向同性、空 coarse 无效零 prompt 与无效 ROI 精确回退 no-mask base 五项契约。",
      ["test", "unit-test", "dense-prompt"], "moderate"),
    CL("scripts/ablations/test_dense_prompt_utils.py", "DensePromptUtilsTest", [21, 107],
       "五个 unittest 用例类，逐一验证 dense prompt 工具的梯度控制与数值契约。",
       ["test", "unittest", "dense-prompt"], "moderate"),

    F("scripts/ablations/validate_ablation_contract.py", "validate_ablation_contract.py",
      "消融实验契约验证器：加载 config 解析 architecture_id 与 wrapper 期望值比对，逐项断言 neck 类型、embedding stride、final-mask 坐标/损失模式、ROI-SAM、P2 refiner、point warmup、shape miner 与 dense prompt 前向数值行为，并实建模型做张量探针，防止 shell 开关与 config 半开错配。",
      ["validation", "ablation", "architecture-contract", "testing"], "complex",
      "main 单函数约 786 行，采用 require() 快速失败断言风格串起全部检查。"),
    FN("scripts/ablations/validate_ablation_contract.py", "parse_args", [16, 57],
       "定义契约验证 CLI：期望 architecture_id、neck、prompt route、explicit mode、final-mask/损失模式、ROI-SAM 与 P2 refiner 开关等。",
       ["cli", "argument-parsing", "validation"], "moderate"),
    FN("scripts/ablations/validate_ablation_contract.py", "main", [69, 854],
       "执行完整契约验证流程：Config 解析 → architecture_id 匹配 → cfg 逐键断言 → 实建模型并做 forward 探针（含 Gaussian 前向、no-mask 回退与 ROI-SAM 裁剪几何）。",
       ["validation", "model-probe", "contract"], "complex"),

    F("scripts/ablations_uecoco/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh",
      "r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh",
      "ue_coco 上的 R1-C4-RD 消融自包含启动脚本：复刻 WHU R1-C4-RD 架构（PAFPN neck、stride-16/64x64 embedding、points_box_dense、detach raw-logits dense prompt、无 P2 refiner），先跑 validate_ablation_contract 再用 torchrun 后台启动训练，数据使用防泄漏 clean split，原生 512 图像 resize 到 1024 以对齐 WHU 的 embedding 几何。",
      ["script", "ablation", "launcher", "ue-coco", "experiment-runner"], "moderate",
      "通过环境变量矩阵加 torchrun 长命令数组组装，支持 TRAINER_EXTRA_ARGS 透传额外训练参数。"),
    F("scripts/ablations_uecoco/wait_then_launch_r1_c4_rd.sh", "wait_then_launch_r1_c4_rd.sh",
      "GPU 空闲监视脚本：轮询 train_rsprompter 进程直至全部退出，再等 15 秒释放显存后以 detached 方式在 0-3 号卡启动 ue_coco R1-C4-RD 实验，事件写入日志。",
      ["script", "scheduler", "launcher", "monitoring"], "simple"),

    F("scripts/resplit_ue_coco.py", "resplit_ue_coco.py",
      "按网格单元重切 ue_coco 以消除区域泄漏：原 split 把同一地理网格 (gx,gy) 的采样点分给不同 split（81/84 格跨 split），本脚本把整格分配到新 train/val/test（目标 70/15/15），重编号图像与标注并写出 *.clean.json，原始文件首次运行时备份为 *.orig.json。",
      ["data-pipeline", "dataset", "leakage-fix", "preprocessing"], "moderate"),
    FN("scripts/resplit_ue_coco.py", "load_all_images", [47, 66],
       "跨三个原始 split 聚集全部图像记录及其网格坐标。",
       ["data-loading", "dataset", "ue-coco"], "simple"),
    FN("scripts/resplit_ue_coco.py", "assign_cells", [69, 95],
       "用固定种子随机打乱网格单元并按 70/15/15 比例把整格分配到 train/val/test。",
       ["split", "randomization", "leakage-fix"], "moderate"),
    FN("scripts/resplit_ue_coco.py", "renumber", [98, 114],
       "按新 split 重排图像与 annotation id 并改写 file_name 引用，保持 COCO 结构一致。",
       ["renumbering", "coco", "split"], "simple"),
    FN("scripts/resplit_ue_coco.py", "main", [117, 195],
       "CLI 主流程：读取标注 → 聚集图像 → 分配网格 → 重编号 → 写出 clean split（支持 --dry-run 只报告）。",
       ["entry-point", "data-pipeline", "cli"], "moderate"),

    F("tools/mask_to_coco_whu512.py", "mask_to_coco_whu512.py",
      "把 WHU-512 二值掩码标签转成 COCO 实例分割 JSON 的数据准备工具：逐连通域提取多边形轮廓（最小面积过滤），为每个 split 生成 images/annotations/categories。",
      ["data-pipeline", "coco", "preprocessing", "dataset"], "moderate"),
    FN("tools/mask_to_coco_whu512.py", "mask_to_polygons", [20, 32],
       "对单个二值掩码做 findContours 并按 min_area 过滤，输出 COCO 多边形坐标列表。",
       ["contour", "mask", "polygon"], "simple"),
    FN("tools/mask_to_coco_whu512.py", "convert_split", [35, 116],
       "转换单个 split：遍历图像、配对掩码、累积多边形标注并写出 COCO JSON。",
       ["conversion", "coco", "mask"], "moderate"),
    FN("tools/mask_to_coco_whu512.py", "main", [119, 134],
       "解析 --data-root/--output-dir 并对 train/val/test 三个 split 执行转换。",
       ["entry-point", "cli", "data-pipeline"], "simple"),

    F("train/__init__.py", "__init__.py",
      "train 包的空 __init__.py 占位文件，使 train/ 成为可导入 Python 包（主训练入口 train_rsprompter_fusion.py 位于其中）。",
      ["package-init", "train", "python"], "simple"),

    F("train/train_rsprompter_fusion.py", "train_rsprompter_fusion.py",
      "项目主训练入口（DDP/FSDP 双模式）：按 mmengine config 经 mmdet 注册表构建 RSPrompterAnchor 检测器，执行 warmup+cosine 调度（按 optimizer step 计数）、梯度累积、AMP、EMA 与分组件学习率组优化；训练循环调用 model.loss 得到 loss_dict（检测损失按 epoch 三阶段加权、P2 refiner 损失跨 rank 归约、coarse shape 监督由 config 调度），rank-0 独占全量验证并跑 COCO bbox/segm 评估，按可配置 metric（含加权/复合表达式）保存 verified best checkpoint（内嵌架构契约与 config 快照），支持 --test-only、--init-from 跨架构初始化与早停。",
      ["entry-point", "training", "ddp", "sam2", "prompt-learning"], "complex",
      "约 2000 行的 main() 集中编排训练全流程；空 GT batch 也进入损失计算以保持各 DDP rank 集合通信序列一致。"),
    FN("train/train_rsprompter_fusion.py", "main", [1325, 3346],
       "训练主流程：argparse 参数校验 → 注册表注册 → 固定随机种子 → 初始化 DDP/FSDP（含 gloo 控制组）→ Config.fromfile 构建模型并解析架构契约 → 数据加载（WHU COCO/labelme、可选 UAV 双流）→ 分组优化器与 warmup+cosine → epoch 循环（损失求和、grad accum、AMP、EMA、非有限损失同步中止）→ rank-0 验证与 COCO 评估 → verified best checkpoint 与早停。",
       ["entry-point", "training-loop", "ddp"], "complex"),
    FN("train/train_rsprompter_fusion.py", "_build_optimizer", [468, 689],
       "按组件学习率倍率（sat backbone、sat other、bbox head、drone、mask decoder、no-mask embed、prompt encoder、shape prior、P2 refiner、quality head、scene align）构建 AdamW 参数组优化器。",
       ["optimizer", "parameter-groups", "lr-mult"], "complex"),
    CL("train/train_rsprompter_fusion.py", "ExponentialMovingAverage", [717, 785],
       "EMA 影子权重类：按 decay/update_every 维护参数与 buffer 的指数滑动平均，支持 apply_to/restore 与 state_dict 序列化，用于验证评估与 best checkpoint 保存。",
       ["ema", "training-infrastructure", "weights"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "build_coco_gt_and_dt", [82, 177],
       "把 GT 与预测实例（含 RLE mask）组装为 pycocotools 的 COCO gt/dt 对象以执行 bbox/segm 评估。",
       ["coco", "evaluation", "serialization"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "run_coco_eval", [180, 197],
       "运行 COCOeval 并汇总为单指标结果字典。",
       ["coco", "evaluation", "metrics"], "simple"),
    FN("train/train_rsprompter_fusion.py", "_init_distributed", [410, 444],
       "解析 RANK/WORLD_SIZE 等环境变量初始化 NCCL 主进程组与独立 gloo 控制组（含超时配置），返回 dist_info。",
       ["ddp", "initialization", "process-group"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_broadcast_stop_training", [447, 461],
       "在控制组上广播 rank-0 的早停/停止标志，保证迟到的控制信号对所有 rank 可见（有独立冒烟测试覆盖）。",
       ["ddp", "control-plane", "broadcast"], "simple"),
    FN("train/train_rsprompter_fusion.py", "_save_verified_best_checkpoint", [242, 288],
       "指标达标时先写临时路径再原子改名地保存 best checkpoint，附带 metric 标签与 config 快照，避免写坏已有最优权重。",
       ["checkpoint", "atomic-write", "best-model"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_save_checkpoint", [1083, 1123],
       "常规 checkpoint 保存：模型（去 DDP 包装）、优化器、调度器、scaler、EMA 与训练历史一并序列化。",
       ["checkpoint", "serialization", "training"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_load_checkpoint", [1126, 1170],
       "恢复训练：载入权重与优化器/调度器/scaler/EMA 状态，并按架构契约校验 checkpoint 一致性。",
       ["checkpoint", "resume", "training"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_load_init_checkpoint", [1176, 1229],
       "仅初始化权重加载（--init-from）：支持前缀排除与显式跨架构迁移（经 ckpt 迁移与契约豁免）。",
       ["checkpoint", "init-from", "transfer"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_materialize_p2_boundary_refiner_loss", [1284, 1322],
       "把各 rank 本地累积的 P2 boundary refiner 损失做跨 rank 归约并物化为 loss_p2_boundary_refiner 损失项。",
       ["loss", "ddp", "all-reduce"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_extract_instances_numpy", [291, 350],
       "从验证预测的 InstanceData 提取 scores/labels/bboxes/masks，mask 可选 RLE 编码，供 COCO 评估使用。",
       ["predictions", "postprocessing", "coco"], "moderate"),
    FN("train/train_rsprompter_fusion.py", "_masks_to_rles", [45, 71],
       "批量把 numpy 二值掩码编码为 COCO RLE。",
       ["rle", "mask", "serialization"], "simple"),
    FN("train/train_rsprompter_fusion.py", "_log_trainable_state", [808, 843],
       "按顶层模块统计并打印可训练参数量，确认冻结策略（如 quality-head-only）生效。",
       ["logging", "diagnostics", "freeze"], "simple"),
]

# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------
def E(source, target, etype, weight):
    return OrderedDict(source=source, target=target, type=etype,
                       direction="forward", weight=weight)

def contains_edges(nodes):
    """file:<p> -> sub-node edges for every function/class node."""
    out = []
    for n in nodes:
        if n["type"] in ("function", "class"):
            out.append(E(f"file:{n['filePath']}", n["id"], "contains", 1.0))
    return out

part1_edges = contains_edges(part1_nodes)
part2_edges = contains_edges(part2_nodes)

# exports (public API only; underscore-private helpers skipped except when
# imported cross-file)
part1_exports = [
    ("configs/_sam2_registry.py", "SAM2ModelConfig"),
    ("configs/_sam2_registry.py", "build_sam2_cfg"),
    ("configs/_sam2_registry.py", "sha256_of_file"),
    ("inference/infer_from_checkpoint.py", "main"),
    ("inference/infer_from_checkpoint.py", "parse_args"),
    ("rsprompter/architecture_contract.py", "file_sha256"),
    ("rsprompter/architecture_contract.py", "architecture_fingerprint"),
    ("rsprompter/architecture_contract.py", "architecture_id"),
    ("rsprompter/architecture_contract.py", "architecture_contract"),
    ("rsprompter/architecture_contract.py", "assert_checkpoint_architecture"),
]
for path, name in part1_exports:
    part1_edges.append(E(f"file:{path}", f"function:{path}:{name}" if name != "SAM2ModelConfig"
                         else f"class:{path}:{name}", "exports", 0.8))
part1_edges.append(E("file:rsprompter/ckpt_migration.py",
                     "function:rsprompter/ckpt_migration.py:migrate_legacy_shape_gate_state_dict",
                     "exports", 0.8))

part2_exports = [
    ("rsprompter/dense_prompt_utils.py", "function", "transform_coarse_prompt"),
    ("rsprompter/dense_prompt_utils.py", "function", "gaussian_prompt_from_roi_coarse"),
    ("rsprompter/dense_prompt_utils.py", "function", "replace_invalid_dense_with_base"),
    ("rsprompter/dense_prompt_utils.py", "function", "paste_roi_to_full_canvas"),
    ("scripts/ablations/smoke_ddp_control_plane.py", "function", "main"),
    ("scripts/ablations/test_dense_prompt_utils.py", "class", "DensePromptUtilsTest"),
    ("scripts/ablations/validate_ablation_contract.py", "function", "parse_args"),
    ("scripts/ablations/validate_ablation_contract.py", "function", "main"),
    ("scripts/resplit_ue_coco.py", "function", "load_all_images"),
    ("scripts/resplit_ue_coco.py", "function", "assign_cells"),
    ("scripts/resplit_ue_coco.py", "function", "renumber"),
    ("scripts/resplit_ue_coco.py", "function", "main"),
    ("tools/mask_to_coco_whu512.py", "function", "mask_to_polygons"),
    ("tools/mask_to_coco_whu512.py", "function", "convert_split"),
    ("tools/mask_to_coco_whu512.py", "function", "main"),
    ("train/train_rsprompter_fusion.py", "function", "main"),
    ("train/train_rsprompter_fusion.py", "function", "build_coco_gt_and_dt"),
    ("train/train_rsprompter_fusion.py", "function", "run_coco_eval"),
    ("train/train_rsprompter_fusion.py", "class", "ExponentialMovingAverage"),
    ("train/train_rsprompter_fusion.py", "function", "_broadcast_stop_training"),
]
for path, kind, name in part2_exports:
    part2_edges.append(E(f"file:{path}", f"{kind}:{path}:{name}", "exports", 0.8))

# imports — MUST equal batchImportData sums (2 total)
part2_edges.append(E(f"file:{SMOKE}", f"file:{TRAIN}", "imports", 0.7))
part2_edges.append(E(f"file:{TEST}", f"file:{DPU}", "imports", 0.7))

# calls (cross-file, grounded in source)
part1_calls = [
    (f"function:{INF}:_resolve_model_config", f"function:{ARC}:architecture_contract"),
    (f"function:{INF}:_resolve_model_config", f"function:{ARC}:file_sha256"),
]
for s, t in part1_calls:
    part1_edges.append(E(s, t, "calls", 0.8))

part2_calls = [
    (f"function:{SMOKE}:main", f"function:{TRAIN}:_broadcast_stop_training"),
    (f"class:{TEST}:DensePromptUtilsTest", f"function:{DPU}:transform_coarse_prompt"),
    (f"class:{TEST}:DensePromptUtilsTest", f"function:{DPU}:gaussian_prompt_from_roi_coarse"),
    (f"class:{TEST}:DensePromptUtilsTest", f"function:{DPU}:replace_invalid_dense_with_base"),
    (f"function:{TRAIN}:main", f"function:{ARC}:architecture_contract"),
    (f"function:{TRAIN}:main", f"function:{ARC}:file_sha256"),
    (f"function:{VAL}:main", f"function:{ARC}:architecture_contract"),
]
for s, t in part2_calls:
    part2_edges.append(E(s, t, "calls", 0.8))

# tested_by (production -> test)
part2_tested_by = [
    (f"file:{DPU}", f"file:{TEST}"),
    (f"file:{DPU}", "file:scripts/smoke_test_components.py"),  # neighborMap cross-batch
    (f"file:{TRAIN}", f"file:{SMOKE}"),
]
for s, t in part2_tested_by:
    part2_edges.append(E(s, t, "tested_by", 0.5))

# depends_on — mmengine _base_ inheritance chains + dynamic loads + launchers
part1_depends = [
    ("configs/whu1024_baseplus_clean.py", "configs/rsprompter_anchor_satS_v11_sam2_large_full.py"),
    ("configs/whu1024_baseplus_explicit_coarse.py", "configs/whu1024_baseplus_clean.py"),
    ("configs/whu1024_baseplus_explicit_coarse_densefix.py", "configs/whu1024_baseplus_explicit_coarse.py"),
    ("configs/rsprompter_anchor_satS_v11_sam2_large_full.py", "configs/_sam2_registry.py"),
    ("inference/infer_from_checkpoint.py", "rsprompter/__init__.py"),
    ("inference/infer_from_checkpoint.py", "rsprompter/architecture_contract.py"),
]
for s, t in part1_depends:
    part1_edges.append(E(f"file:{s}", f"file:{t}", "depends_on", 0.6))

part2_depends = [
    (WAIT, R1C4),
    (R1C4, TRAIN),
    (R1C4, VAL),
    (TRAIN, "rsprompter/__init__.py"),
    (TRAIN, ARC),
    (TRAIN, "data/__init__.py"),
    (VAL, DPU),
    (VAL, "rsprompter/__init__.py"),
    (VAL, ARC),
    (VAL, "data/__init__.py"),
]
for s, t in part2_depends:
    part2_edges.append(E(f"file:{s}", f"file:{t}", "depends_on", 0.6))

# configures — configs to trainer/inference, env script, launcher -> config,
# and configs -> assembled model components (cross-batch files verified on disk)
part1_configures = [
    ("configs/rsprompter_anchor_satS_v11_sam2_large_full.py", TRAIN),
    ("configs/whu1024_baseplus_clean.py", TRAIN),
    ("configs/whu1024_baseplus_explicit_coarse.py", TRAIN),
    ("configs/whu1024_baseplus_explicit_coarse_densefix.py", TRAIN),
    ("configs/whu1024_baseplus_explicit_coarse.py", INF),
    ("configs/environment.sh", TRAIN),
    ("configs/environment.sh", "configs/_sam2_registry.py"),
    ("configs/rsprompter_anchor_satS_v11_sam2_large_full.py", "rsprompter/models.py"),
    ("configs/rsprompter_anchor_satS_v11_sam2_large_full.py", "rsprompter/models_sam2.py"),
    ("configs/whu1024_baseplus_clean.py", "rsprompter/models.py"),
    ("configs/whu1024_baseplus_clean.py", "rsprompter/models_sam2.py"),
    ("configs/whu1024_baseplus_explicit_coarse.py", "rsprompter/models_sam2.py"),
    ("configs/whu1024_baseplus_explicit_coarse.py", "rsprompter/shape_prior.py"),
    ("configs/whu1024_baseplus_explicit_coarse.py", "rsprompter/coarse_mask_loss.py"),
    ("configs/whu1024_baseplus_explicit_coarse.py", "rsprompter/p2_boundary_refiner.py"),
]
for s, t in part1_configures:
    part1_edges.append(E(f"file:{s}", f"file:{t}", "configures", 0.6))
part2_edges.append(E(f"file:{R1C4}", "file:configs/whu1024_baseplus_explicit_coarse.py",
                     "configures", 0.6))

# related
part1_edges.append(E("file:inference/__init__.py", f"file:{INF}", "related", 0.5))
part2_related = [
    (R1C4, RESPLIT),
    (M2C, "configs/whu1024_baseplus_clean.py"),
]
for s, t in part2_related:
    part2_edges.append(E(f"file:{s}", f"file:{t}", "related", 0.5))

# ---------------------------------------------------------------------------
# Validation + write
# ---------------------------------------------------------------------------
BATCH_IMPORT_TOTAL = 2  # sum of batchImportData lengths for this batch

def validate(nodes, edges, part_name, extra_ok_files):
    ids = set()
    for n in nodes:
        assert n["id"] not in ids, f"duplicate node id {n['id']}"
        ids.add(n["id"])
        for field in ("id", "type", "name", "summary", "tags", "complexity"):
            assert n.get(field), f"missing {field} on {n['id']}"
        assert 3 <= len(n["tags"]) <= 5, f"tag count on {n['id']}"
        assert n["complexity"] in ("simple", "moderate", "complex")
    seen_edges = set()
    import_edges = 0
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        assert key not in seen_edges, f"duplicate edge {key}"
        seen_edges.add(key)
        assert e["source"] != e["target"], f"self-referencing edge {key}"
        assert e["source"] in ids, f"{part_name}: edge source not a part node: {e['source']}"
        assert e["direction"] == "forward"
        if e["type"] == "imports":
            import_edges += 1
        # target must be a part node, a cross-part batch node, or a verified
        # cross-batch file/function reference
        if e["target"] not in ids:
            assert e["target"] in extra_ok_files, \
                f"{part_name}: unresolvable edge target {e['target']}"
    return import_edges

# cross-part + verified cross-batch targets
CROSS = {
    "file:train/train_rsprompter_fusion.py",
    "file:rsprompter/__init__.py",
    "file:rsprompter/architecture_contract.py",
    "file:inference/infer_from_checkpoint.py",
    "file:configs/whu1024_baseplus_explicit_coarse.py",
    "file:configs/whu1024_baseplus_clean.py",
    "file:configs/_sam2_registry.py",
    "file:configs/rsprompter_anchor_satS_v11_sam2_large_full.py",
    "file:rsprompter/models.py",
    "file:rsprompter/models_sam2.py",
    "file:rsprompter/shape_prior.py",
    "file:rsprompter/coarse_mask_loss.py",
    "file:rsprompter/p2_boundary_refiner.py",
    "file:rsprompter/dense_prompt_utils.py",
    "file:data/__init__.py",
    "file:scripts/smoke_test_components.py",
    "file:scripts/ablations/validate_ablation_contract.py",
    "file:scripts/ablations/smoke_ddp_control_plane.py",
    "file:scripts/ablations/test_dense_prompt_utils.py",
    "file:scripts/resplit_ue_coco.py",
    "function:rsprompter/architecture_contract.py:architecture_contract",
    "function:rsprompter/architecture_contract.py:file_sha256",
    "function:train/train_rsprompter_fusion.py:_broadcast_stop_training",
    "class:configs/_sam2_registry.py:SAM2ModelConfig",
}

i1 = validate(part1_nodes, part1_edges, "part1", CROSS)
i2 = validate(part2_nodes, part2_edges, "part2", CROSS)
assert i1 + i2 == BATCH_IMPORT_TOTAL, f"imports mismatch: {i1}+{i2} != {BATCH_IMPORT_TOTAL}"

node_count = len(part1_nodes) + len(part2_nodes)
edge_count = len(part1_edges) + len(part2_edges)
parts = math.ceil(max(node_count / 60, edge_count / 120))
assert parts == 2, f"unexpected part count {parts}"
assert len(part1_nodes) <= 60 and len(part1_edges) <= 120
assert len(part2_nodes) <= 60 and len(part2_edges) <= 120

for k, fragment in ((1, (part1_nodes, part1_edges)), (2, (part2_nodes, part2_edges))):
    out = os.path.join(OUT_DIR, f"batch-9-part-{k}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"nodes": fragment[0], "edges": fragment[1]}, fh,
                  ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(fragment[0])} nodes, {len(fragment[1])} edges")

print(f"TOTAL: {node_count} nodes, {edge_count} edges, imports edges={i1 + i2}")
