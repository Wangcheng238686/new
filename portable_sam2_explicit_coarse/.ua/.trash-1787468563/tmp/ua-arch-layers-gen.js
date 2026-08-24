const fs = require('fs');
const input = JSON.parse(fs.readFileSync('/home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/tmp/ua-arch-input.json', 'utf8'));
const ids = new Set(input.fileNodes.map(n => n.id));

function pick(filterFn) { return input.fileNodes.filter(filterFn).map(n => n.id); }

const TEST_FILES = new Set([
  'scripts/smoke_test_components.py', 'scripts/smoke_test_components.sh',
  'scripts/ablations/smoke_all.sh', 'scripts/ablations/smoke_ddp_control_plane.py',
  'scripts/ablations/test_dense_prompt_utils.py', 'scripts/ablations/validate_ablation_contract.py',
]);

// 1. model architecture: rsprompter/
const modelArch = pick(n => n.filePath.startsWith('rsprompter/'));
// 2. data & evaluation: data/, utils/, tools/, scripts/resplit_ue_coco.py
const dataLayer = pick(n => n.filePath.startsWith('data/') || n.filePath.startsWith('utils/')
  || n.filePath.startsWith('tools/') || n.filePath === 'scripts/resplit_ue_coco.py');
// 3. training: train/
const training = pick(n => n.filePath.startsWith('train/'));
// 4. experiment orchestration: scripts/ablations*, scripts/run_whu1024*, scripts/load_environment.sh (minus test files)
const orchestration = pick(n =>
  (n.filePath.startsWith('scripts/ablations') || n.filePath.startsWith('scripts/ablations_uecoco')
   || n.filePath === 'scripts/run_whu1024_explicit_coarse_4gpu.sh' || n.filePath === 'scripts/load_environment.sh')
  && !TEST_FILES.has(n.filePath));
// 5. test & validation
const testLayer = pick(n => TEST_FILES.has(n.filePath));
// 6. config: configs/
const configLayer = pick(n => n.filePath.startsWith('configs/'));
// 7. inference: inference/ + scripts/infer_whu_checkpoint.sh
const inference = pick(n => n.filePath.startsWith('inference/') || n.filePath === 'scripts/infer_whu_checkpoint.sh');

const layers = [
  { id: 'layer:model-architecture', name: '模型架构层',
    description: '基于 SAM2 与 RSPrompter 的提示学习模型核心，包含 mask head、粗掩码监督损失、形状先验点挖掘、P2 边界精修、稠密提示工具，以及 checkpoint 迁移与架构契约校验。',
    nodeIds: modelArch },
  { id: 'layer:data', name: '数据与评估层',
    description: 'WHU 建筑实例、卫星/无人机跨视角数据集的加载与增强，COCO 评估与 mmdet 数据变换工具，以及 COCO 格式转换与 UE-COCO 数据集重划分等预处理脚本。',
    nodeIds: dataLayer },
  { id: 'layer:training', name: '训练层',
    description: 'DDP 分布式训练入口 train_rsprompter_fusion.py，负责融合粗掩码提示学习训练流程的构建与启动。',
    nodeIds: training },
  { id: 'layer:experiment-orchestration', name: '实验编排层',
    description: '四十余个消融实验启动器（coarse points、dense prompt、P2 refiner、ROI 监督等变体）及其 GPU 等待调度、串行监控与环境加载脚本，并附实验矩阵记录 README。',
    nodeIds: orchestration },
  { id: 'layer:test', name: '测试与验证层',
    description: '组件级冒烟测试（smoke test）、DDP 控制面测试、dense prompt 单元测试与消融架构契约校验，用于实验配置的回归验证。',
    nodeIds: testLayer },
  { id: 'layer:config', name: '配置层',
    description: 'mmengine 配置体系，包含 SAM2 模型注册、WHU1024 基线与 explicit-coarse 实验配置、densefix 消融覆盖以及运行环境默认值。',
    nodeIds: configLayer },
  { id: 'layer:inference', name: '推理层',
    description: '从 checkpoint 加载训练模型，对 WHU 数据集执行实例分割推理并输出 COCO 格式结果的入口与封装脚本。',
    nodeIds: inference },
];

// ---- validation ----
const assigned = new Set();
let dup = [];
for (const l of layers) {
  if (l.nodeIds.length === 0) { console.error('EMPTY LAYER:', l.id); process.exit(1); }
  for (const nid of l.nodeIds) {
    if (!ids.has(nid)) { console.error('UNKNOWN ID:', nid); process.exit(1); }
    if (assigned.has(nid)) dup.push(nid);
    assigned.add(nid);
  }
}
const missing = [...ids].filter(i => !assigned.has(i));
if (dup.length || missing.length) {
  console.error('DUP:', dup, 'MISSING:', missing); process.exit(1);
}
const total = layers.reduce((s, l) => s + l.nodeIds.length, 0);
console.log('VALID: layers =', layers.length, ' total assigned =', total, ' expected =', input.fileNodes.length);
layers.forEach(l => console.log(' ', l.id.padEnd(30), l.nodeIds.length));
fs.writeFileSync('/home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/intermediate/layers.json', JSON.stringify(layers, null, 2));
console.log('WROTE /home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/intermediate/layers.json');
