"""Contract, independent-RLE and real batch2 model smoke for the minimal archive."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'train'))
import numpy as np
import torch
from mmengine import Config
from rsprompter.uav.contract import model_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract-only',action='store_true')
    p.add_argument('--data-root',default='/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
    p.add_argument('--gpu',action='store_true');p.add_argument('--output',default=None);args=p.parse_args()
    cfg=Config.fromfile(str(ROOT/'configs/ablations/ue_minimal_single.py'))
    old=Config.fromfile(str(ROOT/'configs/ablations/ue_rgbd_single.py'))
    expected=old.model.to_dict();expected['test_cfg']['rcnn']['max_per_img']=250
    assert expected==cfg.model.to_dict()
    assert cfg.model.roi_head.mask_head.shape_prior_cfg.coarse_mask_output_size==64
    report=dict(contract=model_contract(cfg.model.to_dict()),coarse=64,max_per_img=250,max_dets=[1,10,250])
    if not args.contract_only:
        from data.ue_minimal_dataset import MinimalSatelliteDataset,archive_identity
        from data.loader import rtmdet_collate_fn
        torch.set_num_threads(2)
        ds=MinimalSatelliteDataset(args.data_root,'train');aug=MinimalSatelliteDataset(args.data_root,'train',True)
        counts=[len(s['annotations']) for s in ds.samples]
        # Cover largest train scene plus another image in one batch; no test decoding.
        i=int(np.argmax(counts));j=0 if i!=0 else 1
        a=ds[i];b=ds[j];same=ds[i]
        assert torch.equal(a['img'],same['img']) and np.array_equal(a['gt_masks'],same['gt_masks'])
        transformed=aug[i]
        assert np.array_equal(a['gt_masks'],transformed['gt_masks']) and torch.equal(a['gt_bboxes'],transformed['gt_bboxes'])
        assert transformed['img'].min()>=0 and transformed['img'].max()<=255
        # Every annotation retained, including masks overlapping another instance.
        assert len(a['gt_masks'])==counts[i]
        report.update(archive_hashes=archive_identity(args.data_root),train_count=len(ds),max_train_instances=max(counts),
                      smoke_image_ids=[ds.samples[k]['image_id'] for k in (i,j)],augmentation_preserves_targets=True)
        if args.gpu:
            import rsprompter
            from mmdet.utils import register_all_modules
            register_all_modules(init_default_scope=True)
            from mmdet.registry import MODELS
            from gt_samples import _build_gt_data_samples
            torch.manual_seed(44);torch.backends.cudnn.benchmark=False
            model=MODELS.build(copy.deepcopy(cfg.model)).cuda().train()
            from optimizer import _build_optimizer
            # Forward/backward on the real batch2, including the densest training scene.
            batch=rtmdet_collate_fn([a,b]);imgs=batch['imgs'].cuda()
            samples=_build_gt_data_samples(imgs,batch['img_metas'],[x.cuda() for x in batch['gt_bboxes']],
                    [x.cuda() for x in batch['gt_labels']],batch['gt_masks'],torch.device('cuda'))
            processed=model.data_preprocessor(dict(inputs=imgs,data_samples=samples),training=True)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4)
            scaler=torch.amp.GradScaler('cuda',init_scale=512)
            with torch.autocast('cuda',dtype=torch.float16):
                losses=model.loss(processed['inputs'],processed['data_samples'])
                loss=sum(sum(v) if isinstance(v,list) else v for k,v in losses.items() if 'loss' in k)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            assert torch.isfinite(norm)
            scaler.step(optimizer);scaler.update()
            report.update(batch_size=2,loss=float(loss.detach()),grad_norm=float(norm),peak_train_MiB=torch.cuda.max_memory_allocated()/2**20,
                          finite_update=True)
            model.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                prediction=model.predict(processed['inputs'],processed['data_samples'],rescale=True)
            assert len(prediction)==2 and all(len(x.pred_instances)<=250 for x in prediction)
            report.update(pred_counts=[len(x.pred_instances) for x in prediction],peak_total_MiB=torch.cuda.max_memory_allocated()/2**20)
        # Explicit AP250 numerical calibration with 220 independently overlapping masks.
        from pycocotools.coco import COCO
        from utils.coco_eval_utils import run_coco_eval
        c=COCO();c.dataset=dict(images=[dict(id=1,width=512,height=512)],categories=[dict(id=1,name='building')],
          annotations=[dict(id=k+1,image_id=1,category_id=1,bbox=[(k%22)*20,(k//22)*20,10,10],area=100,iscrowd=0) for k in range(220)])
        c.createIndex();d=c.loadRes([dict(image_id=1,category_id=1,bbox=x['bbox'],score=1-k*.001) for k,x in enumerate(c.dataset['annotations'])])
        metrics=run_coco_eval(c,d,'bbox',max_dets=[1,10,250]);assert abs(metrics['bbox/mAP']-1)<1e-9
        report['AP250_calibration']=metrics['bbox/mAP']
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
