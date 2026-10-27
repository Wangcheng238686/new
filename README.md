# UAV x 卫星实例分割：双流统一架构（UBR）工作区

git worktree（分支 `codex/dual-stream-dev`，权威开发分支）。核心问题一句话：**UAV 信息对卫星侧
实例分割有没有帮助**——统一架构下 dual−single 差异收敛为 {端口⑤ UAV 几何, loss_geometry}。

## 当前状态（2026-10-28）

- **EXP-0028-R2**（级3 单流再封版，UBR v1.1）：在跑，ep 21/100。
- **EXP-0029**（级4 双流·真实臂）：在跑，ep 7/100（与 R2 共租四卡）。
- **EXP-0030**（级4 +RGB 臂）：实现/审计完毕，按 PI 指令冻结待发射。

## 从这里开始

- **[docs/WORKSPACE-MAP.md](docs/WORKSPACE-MAP.md)** —— 工作区总地图：活跃资产、历史结论
  一行谱、数据盘工件地图、训练发射 runbook（含 wrapper-only 纪律）。
- **[AGENTS.md](AGENTS.md)** —— 执行纪律（论文骨架、实验规程、git 规约、代码规范）。
- **[docs/EXP_REGISTRY.md](docs/EXP_REGISTRY.md)** —— 全量实验台账（EXP-0001…0030，权威）。

## 代码布局

`portable_sam2_explicit_coarse/` 为主包：`rsprompter/uav/`（UBR 生产模块与双流接线）、
`configs/ablations/ue_ubr_*.py`（三个活跃臂配置）、`scripts/probes/ue_ubr_*.py`（证据层）、
`scripts/ue/run_ubr_*.sh`（唯一合法发射 wrapper）、`train/train_rsprompter_fusion.py`（宿主训练器）。

## 历史（WHU SurfaceBEV 迁移起点）

本 worktree 源自单流 WHU a3sgm 封装迁移（`fast-whu150@95026ae`），CSPM/UDPR 沿封存实现；
迁移验收记录见 [docs/WHU_SURFBEV_MIGRATION.md](docs/WHU_SURFBEV_MIGRATION.md) 与
`docs/whu_surfacebev_migration/acceptance.json`。旧 P2/P3、A1、C5v2 B0 实现可从归档分支找回；
历史实验文档保持原身份，不作为新基线证据。
