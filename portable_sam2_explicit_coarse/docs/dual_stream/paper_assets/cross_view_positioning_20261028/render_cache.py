from pathlib import Path
import json,hashlib,warnings
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import ConnectionPatch

OUT=Path(__file__).resolve().parent
CACHE=Path('/data/wangcheng/checkpoint/ue_native_sutc_engineering_20261028/view_subset_cache')
DATA=Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021/points/001234')
plt.rcParams.update({'font.family':font_name,'font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42})
sat=np.asarray(Image.open(DATA/'satellite/rgb.png').convert('RGB'))
views=[('1 视角',[0],'views_000'),('3 视角',[0,144,288],'views_000_144_288'),('5 视角',[0,72,144,216,288],'views_000_072_144_216_288')]
rows=[];summary=[]
for title,ids,key in views:
 path=CACHE/key/'001234.npz'
 with np.load(path,allow_pickle=False) as z:
  d={k:z[k].copy() for k in ('fields','support','index','anchor')}
 assert d['support'].shape==(512,512)
 d['support']=d['support'].astype(bool)
 d['height']=d['fields'][0].astype(float)*100
 d['count']=np.rint(d['fields'][3].astype(float)*5).astype(int)
 d['density']=(d['index']>=0).sum(1).reshape(128,128)
 rows.append(d)
 summary.append(dict(view_ids=ids,support_pixels=int(d['support'].sum()),support_fraction=float(d['support'].mean()),candidates=int(d['density'].sum()),height_metres_min=float(d['height'][d['support']].min()),height_metres_max=float(d['height'][d['support']].max()),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
fig,axes=plt.subplots(3,4,figsize=(13.6,11.9))
fig.subplots_adjust(left=.105,right=.98,top=.91,bottom=.14,wspace=.08,hspace=.12)
heads=['卫星 RGB + 有效观测范围','最近观测表面的相对高度','每格保留的 UAV 候选数','近表面有效观察视角数']
for j,h in enumerate(heads):axes[0,j].set_title(h,pad=13,fontsize=12)
height_cmap=plt.get_cmap('viridis').copy();height_cmap.set_bad('#d9d9df')
count_cmap=plt.get_cmap('cividis',5).copy();count_cmap.set_bad('#d9d9df')
ims=[]
for i,(d,(title,ids,key)) in enumerate(zip(rows,views)):
 s=d['support']
 overlay=sat.copy();overlay[~s]=[197,190,215]
 axes[i,0].imshow(overlay,interpolation='nearest')
 a=axes[i,1].imshow(np.ma.array(d['height'],mask=~s),cmap=height_cmap,vmin=-5,vmax=120,interpolation='nearest')
 b=axes[i,2].imshow(d['density'],cmap='magma',vmin=0,vmax=8,interpolation='nearest',extent=(-.5,511.5,511.5,-.5))
 c=axes[i,3].imshow(np.ma.array(d['count'],mask=~s),cmap=count_cmap,norm=BoundaryNorm(np.arange(.5,6),5),interpolation='nearest')
 ims=[a,b,c]
 axes[i,0].text(-.13,.5,title+'\n'+(('/'.join(map(str,ids)))+'°' if len(ids)<5 else '0/72/144°\n216/288°'),transform=axes[i,0].transAxes,ha='right',va='center',fontsize=12)
 axes[i,0].text(.02,.02,'覆盖 %.1f%%'%(s.mean()*100),transform=axes[i,0].transAxes,ha='left',va='bottom',color='white',bbox={'facecolor':'black','alpha':.7,'pad':3,'edgecolor':'none'},fontsize=11)
 axes[i,2].text(.02,.02,'%s 个候选'%format(int(d['density'].sum()),','),transform=axes[i,2].transAxes,ha='left',va='bottom',color='white',bbox={'facecolor':'black','alpha':.7,'pad':3,'edgecolor':'none'},fontsize=11)
 for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
for j,(im,label,ticks) in enumerate(zip(ims,['米；相对采样原点','候选数 / 卫星 4×4 像素格','个；距最近观测表面 ≤1 m'],[[-5,40,80,120],[0,2,4,6,8],[1,2,3,4,5]]),1):
 pos=axes[-1,j].get_position();cax=fig.add_axes([pos.x0,.087,pos.width,.015])
 cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ticks);cb.set_label(label,fontsize=10)
fig.suptitle('UAV 几何缓存实际内容：同一训练样本 001234 的 1 / 3 / 5 视角',fontsize=16,y=.974)
fig.text(.105,.936,'所有图均在卫星坐标下；三组使用相同色标，没有根据每张图自动拉伸。',fontsize=11,color='#444444')
fig.text(.105,.025,'紫灰覆盖区 / 灰色高度区：无观测支持；不是高度为零。候选图为 128²，其他空间图为 512²。\n这里只展示固定几何预处理结果，不是预测掩码、三维重建质量或 AP 证据。',fontsize=10,color='#444444')
with warnings.catch_warnings(record=True) as recorded:
 warnings.simplefilter('always');fig.canvas.draw()
 missing=[str(w.message) for w in recorded if 'Glyph' in str(w.message)]
 assert not missing,missing
fig.savefig(OUT/'cache_overview.png',dpi=150,facecolor='white')
fig.savefig(OUT/'cache_overview.pdf',facecolor='white')
plt.close(fig)

# Explain the index/anchor table with actual retained correspondences, not a new warp.
d=rows[-1];index=d['index'];anchor=d['anchor']
valid=(index>=0)&(index//16384==0)
cells=np.where(valid.any(1))[0]
rng=np.random.default_rng(44)
# Spatially spread examples: choose one candidate near each predeclared target location.
locations=[(80,80),(256,80),(432,80),(80,256),(432,256),(80,432),(256,432),(432,432)]
chosen=[]
for x,y in locations:
 best=np.argmin((cells%128*4+2-x)**2+(cells//128*4+2-y)**2)
 cell=int(cells[best]);slot=int(np.where(valid[cell])[0][0]);chosen.append((cell,slot))
fig,axs=plt.subplots(1,2,figsize=(11.8,6.4))
fig.subplots_adjust(left=.035,right=.97,top=.86,bottom=.13,wspace=.13)
axs[0].imshow(sat);axs[0].set_title('卫星原始 RGB：投影目标坐标',fontsize=13)
axs[1].imshow(Image.open(DATA/'uav/yaw_000/rgb.png'));axs[1].set_title('UAV yaw=0° 原始 RGB：源像素坐标',fontsize=13)
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#F0E442','#111111']
points=[]
for n,((cell,slot),color) in enumerate(zip(chosen,colors),1):
 a=anchor[cell,slot];satxy=a[3:5];srcxy=a[:2]
 for ax,xy in zip(axs,(satxy,srcxy)):
  ax.scatter(*xy,s=145,c=[color],edgecolors='white',linewidths=1.3,zorder=5)
  ax.annotate(str(n),xy,xytext=(7,-9),textcoords='offset points',fontsize=12,color='black',bbox=dict(facecolor='white',alpha=.9,pad=1,edgecolor='none'),zorder=7)
 con=ConnectionPatch(xyA=satxy,xyB=srcxy,coordsA='data',coordsB='data',axesA=axs[0],axesB=axs[1],color=color,alpha=.6,linewidth=.9,zorder=2)
 fig.add_artist(con)
 points.append(dict(label=n,cell=cell,candidate_slot=slot,source_token=int(index[cell,slot]),source_uv=srcxy.tolist(),satellite_uv=satxy.tolist(),source_camera_Z_m=float(a[2]),satellite_camera_Z_m=float(a[5])))
for ax in axs:ax.set_xlim(-.5,511.5);ax.set_ylim(511.5,-.5);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('index / anchor 在存什么？八个实际保留的几何对应示例',fontsize=15,y=.97)
fig.text(.035,.035,'同号点是一条缓存对应：UAV 源像素 + 深度，经内外参投影到卫星坐标。\n示例来自五视角缓存中的 yaw=0° 候选；这是几何对应展示，不认证无遮挡或跨视角外观完全一致。',fontsize=11,color='#444444')
fig.savefig(OUT/'anchor_correspondences.png',dpi=150,facecolor='white')
fig.savefig(OUT/'anchor_correspondences.pdf',facecolor='white')
plt.close(fig)
(OUT/'visualization_data.json').write_text(json.dumps(dict(image_id=1234,summary=summary,example_anchors=points),indent=2)+'\n')
print(json.dumps(summary,indent=2))
