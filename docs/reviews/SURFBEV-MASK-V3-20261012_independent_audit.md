# SURFBEV-MASK-V3-20261012 独立审计

## 结论

**数据物化与规则契约 PASS。** 独立审计员只读复算全部 1,200 场，逐像素错误为 0。
本审计不把自动规则等同于完整建筑标注质量，也不提供 AP 证据。

## 核查范围与结果

- 输入 pair/split manifest 分别固定为 SHA256 `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`、`17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5`。
- 1,200 个唯一场景、1,200 个派生 PNG；train/val/test/excluded 为 694/100/151/255。
- 每场独立重算原始 512² ID 的面积、bbox、触边、规则命中 ID；与 manifest 完全一致。原始及派生 PNG 的当前 SHA256 均与 manifest 一致。
- 逐像素差异恰好是命中 local ID 的所有像素变成 0；未命中 ID、其像素及 RGB 不变；派生标签没有残余命中 ID。
- 删除总计 1,072 实例、21,360 像素：train 569/11,584，val 44/330，test 165/5,458，excluded 294/3,988。
- test 与其他 split 使用同一规则，但没有进入 48 场 train/val 可视化抽样，也没有参与阈值制定。抽样四层（train/val × 变更/未变更）各 12 场，无重复。
- 脚本只新写派生 labels、manifest、summary、spotcheck；未写原始标签或模型侧 `IGNORE`。当前源哈希一致可证当前产物与来源关系，不能单靠这次产物追证更早历史文件的绝对不变性。

## 限制

上述结论仅是实现与契约审计。是否误删完整可辨的小楼，仍依赖预注册的抽样视觉检查；该检查记录见 `docs/SURFBEV-MASK-V3-20261012_conclusion.md`。
