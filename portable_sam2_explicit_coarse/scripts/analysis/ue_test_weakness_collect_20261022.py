"""Consolidate independent CPU weakness audits; never select a new model route."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=Path('/data/wangcheng/checkpoint/ue_test_weakness_20261022')
DOC=ROOT/'docs/dual_stream'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(n,d):return 100*n/d if d else 0

def main():
 d=read(R/'distribution/audit.json');e=read(R/'errors/summary.json');s=read(R/'support/summary.json')
 assert d['analysis_source_sha256']==sha(ROOT/'scripts/analysis/ue_test_distribution_audit_20261022.py')
 assert e['script_sha256']==sha(ROOT/'scripts/analysis/ue_test_error_modes_20261022.py')
 assert s['protocol']['script_sha256']==sha(ROOT/'scripts/analysis/ue_test_support_audit_20261022.py')
 assert s['protocol']['images']==e['images']==459
 assert s['all']['instances']==44988
 assert sum(v['instances'] for v in s['true_mask_area_groups'].values())==44988
 assert sum(v['instances'] for v in s['native_coverage_groups'].values())==44988
 for p,h in d['sources_sha256'].items():assert sha(p)==h,p
 for spec in e['sources'].values():assert sha(spec['path'])==spec['sha256']
 manifest=read(R/'manifest.json')
 for path,spec in manifest['source_files'].items():assert sha(path)==spec['sha256']
 for n,h in s['protocol']['record_sha256'].items():
  name={'baseline':'baseline','native':'native_geometry','unit':'unit'}[n]
  assert h==e['sources'][name]['sha256']
 for kind in ('bbox','segm'):
  for v in e['coco_matching'][kind].values():assert v['gt']==44988
 assert 'test' in d['baseline']
 for split in ('val','test'):
  for kind in ('segm','bbox'):
   assert abs(d['baseline'][split][kind]['all']['metrics']['AP']-d['baseline'][split]['offline_true_mask_area_grouping'][kind]['AP'])<1e-12
 h=e['hungarian'];assert h['bbox_matched']+h['gt_unmatched']==h['gt']==44988
 assert h['bbox_matched']+h['dt_unmatched']==h['dt']
 sub=read(R/'support/selection_subset.json');assert len(sub['images'])==32
 subset={}
 for zone,den in [('gt','gt_pixels'),('boundary2','boundary_pixels')]:
  n=sum(a[den] for a in sub['images']);subset[zone]=dict(pixels=n,
   **{k:sum(a[zone+'_'+k] for a in sub['images'])/n for k in ('all_stride2_points','retained_exact_points','all_stride2_1px','retained_exact_1px')})
 report=['# v1测试集弱点：三路离线审计','',
 '本轮由三个子代理启动，后续因代理额度中断由主代理接手核对与汇总。仅CPU已有数据/预测，未新增训练或GPU推理，未修改本轮冻结探针合同。',
 '测试集已被反复用于开发诊断；本报告是探索性证据，不作为独立最终泛化认证。',
 '', '## 1. 分布与场景','',
 '|split|图数|实例数|平均实例/图|>100实例图|真实mask small比例|触边实例比例|',
 '|---|---:|---:|---:|---:|---:|---:|']
 for name,v in d['split_summary'].items():
  report.append(f"|{name}|{v['images']}|{v['instances']}|{v['mean_instances']:.2f}|{v['density_images'].get('>100',0)}|{pct(v['mask_size_counts']['small'],v['instances']):.2f}%|{pct(v['edge_instances'],v['instances']):.2f}%|")
 report+=['','small按真实mask面积<1024像素，medium<9216，large其余。触边按GT框触及图边代理。',
 'test小实例、密集图及形状填充率差异突出；触边比例反而val更高，不能用test触边更多解释落差。空间中心分布也不同，但中心距离不证明场景资产独立或视野无重叠。',
 '', '|图像密度|val图数 / Mask AP|test图数 / Mask AP|','|---|---|---|']
 for label in ('<=50','51-100','>100'):
  vals=[]
  for split in ('val','test'):
   g=d['baseline'][split]['segm']['density_'+label];m=g['metrics'];vals.append(f"{g['images']} / {100*m['AP']:.3f}" if m else f"{g['images']} / NA")
  report.append('|'+label+'|'+'|'.join(vals)+'|')
 report+=['','同密度组仍有val/test差距，密度不能单独解释全部退化。子组AP是重新累计结果，不能按图数加权还原总AP。机械标签检查未发现RLE面积/长度不符、越界框或同图完全重复RLE；不代表语义标签人工核验。',
 '', '## 2. 检测与掩码错误分开看','',
 f"- 固定bbox IoU≥.5最大匹配数优先Hungarian：{h['bbox_matched']} / {h['gt']} GT有一对一框，{h['gt_unmatched']}未配对（{pct(h['gt_unmatched'],h['gt']):.2f}%）；其中{h['gt_no_any_bbox_iou50']}个GT根本没有任何最终框达到.5。",
 f"- 已匹配框中，mask IoU≥.75：{h['matched_mask_pass_0.75']} / {h['bbox_matched']}；≥.90：{h['matched_mask_pass_0.9']} / {h['bbox_matched']}，仍有高质量掩码空间。",
 f"- small未匹配框：{e['groups']['size/small']['gt']-e['groups']['size/small']['bbox_matched']} / {e['groups']['size/small']['gt']}；全部未配对GT中small占{pct(e['groups']['size/small']['gt']-e['groups']['size/small']['bbox_matched'],h['gt_unmatched']):.2f}%。",
 '- 最终DT最多193/图，0/459触及250上限；不是maxDets250截断。只能用最终记录确认现象，不能定位RPN/NMS/score阈值的因果贡献。',
 '- bbox一对一匹配失败不是掩码IoU绝对上限，也不等于纯漏检；框不足/定位/竞争均可能参与。',
 '', '### 现有分支具体改了哪些错误','',
 '|分支|IoU .90纠错/新增|IoU .95纠错/新增|边界像素纠错/新增|背景像素纠错/新增|','|---|---:|---:|---:|---:|']
 for n in ('native_geometry','unit'):
  c=e['fixed_pair_changes'][n];p=e['pixel_changes'][n]
  report.append(f"|{n}|{c['corrected_0.9']}/{c['new_error_0.9']}|{c['corrected_0.95']}/{c['new_error_0.95']}|{p['boundary_corrected']}/{p['boundary_new_errors']}|{p['background_corrected']}/{p['background_new_errors']}|")
 report+=['','分母为同一40376个固定框匹配实例，像素区域分母分别见errors/summary.json；实例重叠可重复计像素。这些阈值计数不是AP分解。native背景新增错误多于纠正，表明有益轮廓修改也伴随外扩副作用，不能笼统称全面修好。',
 '错误审计的形态学使用OpenCV默认边界值；支持审计显式将图外视为背景，因此两者的2px边界像素分母不同，不直接相减。',
 '', '## 3. UAV支持与候选压缩','',
 '以下32图固定均匀索引抽样，未按错误挑样；支持是投影观测代理，不是严格可见性认证。',
 '|区域|原stride2点覆盖|保留锚点覆盖|原点1px邻域覆盖|保留锚点1px邻域覆盖|','|---|---:|---:|---:|---:|']
 for z,v in subset.items():report.append('|'+z+'|'+'|'.join(f"{100*v[k]:.2f}%" for k in ('all_stride2_points','retained_exact_points','all_stride2_1px','retained_exact_1px'))+'|')
 report+=['','离散候选压缩显著减少精确落点；1px邻域可恢复较多覆盖，但“附近有特征”不等于精确3D对应。当前B/D只检验已保留锚点局部邻域，不能恢复完全无锚点区。',
 '全459图观测支持/错误分组见support/summary.json；旧float16 geom重建锚点坐标最大实测误差约0.000489px，接近整数边界可能改变栅格化像素，故全量点支持标为近似。32图额外使用原深度重建精确点核对。',
 f"全量按实例等权平均：native GT覆盖{s['all']['native_coverage']*100:.2f}%，候选格覆盖{s['all']['cell_coverage']*100:.2f}%，保留锚点1px邻域覆盖约{s['all']['retained1px_coverage_approx']*100:.2f}%。这三个数有不同空间支持定义，不可当作三种方法的精度。",
 f"基座漏掉的2px内边界前景像素中，native支持覆盖{pct(s['pixel_totals']['boundary2_baseline_missed_native_support_n'],s['pixel_totals']['boundary2_baseline_missed_n']):.2f}%，保留锚点1px邻域覆盖约{pct(s['pixel_totals']['boundary2_baseline_missed_retained1px_n'],s['pixel_totals']['boundary2_baseline_missed_n']):.2f}%。因此不能把多数边界漏前景简单解释为完全没有UAV观测；这也不证明观测内容已正确对齐或足以区分实例。",
 f"按固定native覆盖率组，<.5的1300个已匹配实例native平均IoU差为{s['native_coverage_groups']['<0.5']['native_mean_paired_iou_delta']:+.6f}；≥.8的34833个匹配实例为{s['native_coverage_groups']['>=0.8']['native_mean_paired_iou_delta']:+.6f}。支持质量可能调节收益，但组间差异仍可能混合尺寸/场景，不据此修改本轮门控或阈值。",
 '支持审计采用最大bbox IoU总和再阈值的Hungarian，错误审计先最大化≥.5匹配数；各自固定内部配对，跨报告匹配实例分母不可混用。',
 '', '## 4. 尺寸指标解释缺口','',
 'canonical evaluator将GT area设为bbox宽×高；现有s/m/l是bbox面积按COCO阈值分组，不能称真正mask面积分组。test真实mask-small32400个，bbox-small仅15019个。',
 '已独立将GT area改为mask面积重算分组，整体Mask/bbox AP在val/test数值回放均不变；没有改冻结evaluator，也不覆盖旧产物。这个补算只修正GT面积字段，其他DT/匹配合同沿用原实现。',
 '', '## 5. 等探针完成后的判定表','',
 '|待回答问题|需要合并的证据|可支持的后续讨论|','|---|---|---|',
 '|读取限制是否关键|B−A、D−C及相同内容对照，结合边界点压缩与支持组|只有局部读取净收益成立，才讨论扩大读取能力|',
 '|融合消费是否关键|C−A、D−B、交互，结合常量/零特征容量对照|区分联合消费收益与纯容量收益|',
 '|训练动力是否关键|固定D aux有/无×正确/常量/零，梯度与真实内容干预|辅助IoU或梯度单独不决定方法|',
 '|检测瓶颈是否应纳入|未配对/小实例统计与上述分支完整实例AP|当前冻结框探针不能回答检测增益；待当前研究收敛再决定是否另立检测实验|',
 '', '当前不修改结构、采样范围、轮次、选权或辅助系数；不据测试子组临时追加训练。最终路线待固定E10矩阵及来源/冻结审计完成后再决定。',
 '', '## 产物与核对','',f'产物目录：`{R}`。',
 'distribution/audit.json、errors/summary.json、support/summary.json均记录来源与脚本SHA；主代理已核对共同GT/记录来源、样本数、匹配计数恒等式和总体AP面积回放不变。',
 '本轮没有额外bootstrap；分层和关联为探索性描述，不替代当前固定主比较。']
 (DOC/'UE_TEST_WEAKNESS_SYNTHESIS_20261022.md').write_text('\n'.join(report)+'\n')
 dump=dict(offline_audits_complete=True,images=459,instances=44988,decision_deferred=True,
  inputs={str(p.relative_to(R)):sha(p) for p in [R/'distribution/audit.json',R/'errors/summary.json',R/'support/summary.json',R/'support/selection_subset.json']},
  source_hashes={p.name:sha(p) for p in (ROOT/'scripts/analysis').glob('ue_test_*20261022.py')},
  no_frozen_experiment_changes=True,full_test_point_AP_unchanged_by_GT_area_correction=True)
 (R/'complete_audit.json').write_text(json.dumps(dump,indent=2)+'\n')
 print('COMPLETE',DOC/'UE_TEST_WEAKNESS_SYNTHESIS_20261022.md')

if __name__=='__main__':main()
