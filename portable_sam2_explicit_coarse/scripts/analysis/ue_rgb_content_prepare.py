import json,hashlib,random,math,subprocess
from pathlib import Path
R=Path('/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis')
O=R/'rgb_content_audit';O.mkdir(exist_ok=True)
manifest=R.parent/'dataset/manifest.json';m=json.loads(manifest.read_text());ids=m['splits']['validation'];donor_ids=m['splits']['train'];rows={x['index']:x for x in m['points']}
def eligible(i,j):
 a,b=rows[i],rows[j];p,q=a['footprint_xy_m'],b['footprint_xy_m']
 separated=p[2]<q[0] or q[2]<p[0] or p[3]<q[1] or q[3]<p[1]
 distance=math.dist(a['origin_m'][:2],b['origin_m'][:2])
 return i!=j and not(set(a['building_ids'])&set(b['building_ids'])) and separated and distance>=1500
maps={};previous={}
for name,seed in [('donor_a',1701),('donor_b',1702)]:
 rng=random.Random(seed);choices={i:[j for j in donor_ids if eligible(i,j) and previous.get(i)!=j] for i in ids}
 for v in choices.values():rng.shuffle(v)
 assigned={}
 def match(i,seen):
  for j in choices[i]:
   if j in seen:continue
   seen.add(j)
   if j not in assigned or match(assigned[j],seen):assigned[j]=i;return True
  return False
 order=ids.copy();rng.shuffle(order)
 for i in order:assert match(i,set()),i
 mapping={i:j for j,i in assigned.items()};assert len(mapping)==len(ids) and len(set(mapping.values()))==len(ids)
 records=[]
 for i in ids:
  j=mapping[i];assert eligible(i,j)
  files=[]
  for yaw in (0,72,144,216,288):
   path=Path(m['uav_root'])/rows[j]['uav_dir']/'oblique'/f'view_yaw_{yaw:03d}'/'oblique_rgb.0000.png'
   files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  records.append(dict(target=i,donor=j,origin_distance_m=math.dist(rows[i]['origin_m'][:2],rows[j]['origin_m'][:2]),shared_known_buildings=[],footprint_rectangles_disjoint=True,rgb_files=files))
 maps[name]=dict(seed=seed,records=records);previous=mapping
out=dict(manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),image_ids=ids,donor_pool='frozen manifest train split',maps=maps,scope='Frozen development subset validation only; known satellite building IDs do not enumerate all UAV-visible buildings')
p=O/'donor_maps.json';p.write_text(json.dumps(out,indent=2));print(p,'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
protocol='''# RGB内容干预预注册（运行前）

日期2026-10-15。路由：A类冻结诊断，不改变模型结构/权重，不训练。关联EXP-0006/0007；当前冻结开发子集153验证图，非完整生产开发集或test。

固定dual E72 strict EMA。三臂：正确RGB重新从文件读取的sham；donor_a（seed1701）；donor_b（seed1702）。供体从冻结manifest的627张训练图中选择，153个供体互不重复，无自配对。验证区域集中，在验证内部部分目标没有满足1500m约束的供体，因此运行前改用同一数据冻结版本的训练图RGB。供体可能已被训练看到，属于本轮干预局限；不读取供体标签作为模型输入；两套供体逐目标不同；已知建筑ID无交集、manifest卫星footprint矩形不相交且原点距离≥1500m。建筑ID并非所有UAV可见物体的完整枚举，不能声称绝对没有共同可见建筑。

五视角整体替换，保持yaw顺序、通道、像素尺度；目标卫星RGB/标签、全部深度派生payload（候选、参考点、覆盖、高度、置信、投影等）逐字段hash不变。原始深度在模型中经payload进入，不改深度文件/几何缓存。仅改变payload.rgb。RGB与目标深度不再物理一致，这是受控分布变化，不能作净贡献的因果估计。

主指标：完整验证segm/mAP(correct−donor)，伴随bbox/mAP和三种共同框逐GT mask IoU。固定框沿用上次完整审计的GT框、single预测框、dual预测框；后两者是GT辅助oracle关联，非AP。另按GT bbox面积和237个global建筑等权统计。

选择口径：已固定E72 EMA，无重新选模、无参数更新。正确RGB必须逐图复现历史框/score/mask及共同框IoU；same-file RGB与原payload逐值相等。任一不变量失败立即停止并修复，不解释实验结果。

不确定性：500次配对图像bootstrap，seed1703，对segm/bbox AP差；同一重采样跨模式一致。仅探索性描述，不覆盖空间相关性、模型选优及训练seed方差。两套供体是映射敏感性检查，不是独立训练seed。

诊断门（不作统计显著性/净贡献主张）：两套映射的segm AP均下降≥.005且各探索性95%图像bootstrap区间下界>0，称“稳定的配对RGB内容敏感性”；若三种共同框IoU均下降≥.005（两映射均满足），进一步称“固定框mask路径亦敏感”。两映射segm AP差绝对值均≤.002且三协议IoU差绝对值均≤.002，称“该干预下内容敏感性较弱”。其余为混合结果，分解检测与mask作用，不挑供体报喜。效应线只是本轮工程诊断门。

即使门通过，也只支持当前权重对与几何配对的RGB敏感。确认有效信息净增益需要匹配训练对照；本轮不自动启动训练或修改融合。独立实现及结果审核全文落盘。
'''
(O/'PREREGISTRATION.md').write_text(protocol)
print('maps',[(n,len(x['records']),min(r['origin_distance_m'] for r in x['records'])) for n,x in maps.items()])
