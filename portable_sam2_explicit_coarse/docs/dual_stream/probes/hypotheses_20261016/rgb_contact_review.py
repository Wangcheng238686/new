import json,pathlib
from PIL import Image,ImageDraw
m=json.load(open('/tmp/ue_uav_experiments/20261014_ad_seed42/dataset/manifest.json'))
out=Image.new('RGB',(6*512,3*538),'white');d=ImageDraw.Draw(out)
records=[]
for row,idx in enumerate([431,28,653]):
 p=next(x for x in m['points'] if x['index']==idx)
 paths=[pathlib.Path(m['satellite_root'])/p['satellite_dir']/'satellite_rgb.0000.png']+[pathlib.Path(m['uav_root'])/p['uav_dir']/'oblique'/('view_yaw_%03d'%yaw)/'oblique_rgb.0000.png' for yaw in [0,72,144,216,288]]
 for col,pa in enumerate(paths):
  im=Image.open(pa).convert('RGB');out.paste(im,(col*512,row*538+26));d.text((col*512+8,row*538+8),str(idx)+' '+('sat' if col==0 else 'yaw '+str([0,72,144,216,288][col-1])),fill='black')
 records.append({'index':idx,'files':list(map(str,paths))})
out.save('/tmp/ue_h1_rgb_contact.png');json.dump(records,open('/tmp/ue_h1_rgb_review_sources.json','w'),indent=2)
