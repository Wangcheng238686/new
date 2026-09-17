import sys, json, copy
sys.path.insert(0, '.')
from pathlib import Path
from types import SimpleNamespace
import torch
from inference.infer_from_checkpoint import (_load_checkpoint, _snapshot, _resolve_model_config,
    _register_and_build, _load_model_state, _select_state_dict, _resolve_dataset_contract,
    _build_loader, _build_data_samples, _extract_instances_numpy)
from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval

ckpt_path = sys.argv[1]
thresholds = [float(x) for x in sys.argv[2].split(',')]
split = sys.argv[3] if len(sys.argv) > 3 else "validation"
ckpt = _load_checkpoint(Path(ckpt_path)); snap = _snapshot(ckpt)
ns = SimpleNamespace(checkpoint=ckpt_path, config=None, split="validation", data_root=None,
                     ann_file=None, image_subdir=None, image_size=None, batch_size=2,
                     sam2_repo=None, sam2_ckpt=None, weights="model")
cfg, _ = _resolve_model_config(ns, snap)
contract = _resolve_dataset_contract(ns, snap)
if split == "train":
    # Protocol-clean threshold calibration source: tune on the train split,
    # report on validation.  Same data root / image subdir, ann swap only.
    contract = dict(contract)
    contract["ann_file"] = str(contract["ann_file"]).replace("_val", "_train")
ns.split = split
out = {}
for thr in thresholds:
    c = copy.deepcopy(cfg)
    # test_cfg is top-level in the model config (rcnn dict); also mirror on
    # the built roi_head's live ConfigDict after construction.
    if "test_cfg" in c:
        c["test_cfg"]["rcnn"]["mask_thr_binary"] = thr
    if "test_cfg" in c.get("roi_head", {}):
        c["roi_head"]["test_cfg"]["mask_thr_binary"] = thr
    model = _register_and_build(c)
    st = _load_model_state(model, _select_state_dict(ckpt, "model"), allow_nonstrict=False)
    assert not st["missing_keys"] and not st["unexpected_keys"]
    model.to("cuda:0").eval()
    if hasattr(model.roi_head, "test_cfg") and model.roi_head.test_cfg is not None:
        model.roi_head.test_cfg.mask_thr_binary = thr
    loader = _build_loader(contract, 0)
    cats = VHR10_CATEGORIES if int(contract.get("num_classes", 0) or 0) == 10 else None
    all_gt, all_dt, metas = [], [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, samples = _build_data_samples(batch, torch.device("cuda:0"))
            p = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
            outs = model.predict(p["inputs"], p["data_samples"], rescale=False)
            for i, o in enumerate(outs):
                meta = dict(batch["img_metas"][i]); shape = meta["img_shape"]
                all_gt.append(_extract_instances_numpy(p["data_samples"][i].gt_instances, shape))
                pred = o.pred_instances if hasattr(o, "pred_instances") else o
                all_dt.append(_extract_instances_numpy(pred, shape)); metas.append(meta)
    g, d = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=cats)
    m = run_coco_eval(g, d, iou_type="segm")
    out[thr] = (m["segm/mAP"], m["segm/mAP_75"])
    print(f"thr={thr:.2f} segm/mAP={m['segm/mAP']:.4f} AP75={m['segm/mAP_75']:.4f}", flush=True)
    import json as _json
    print("FULL_METRICS " + _json.dumps({k: float(v) for k, v in m.items()}), flush=True)
    del model; torch.cuda.empty_cache()
print(json.dumps({str(k): v for k, v in out.items()}))
