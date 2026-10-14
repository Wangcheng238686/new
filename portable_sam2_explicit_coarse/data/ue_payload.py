"""One device-transfer contract for train, validation and independent inference."""

import torch
from mmengine.structures import BaseDataElement


def attach_uav_payload(samples, batch):
    if "uav_payloads" not in batch:
        return
    if len(samples) != len(batch["uav_payloads"]):
        raise ValueError("UAV payload batch length mismatch")
    for sample, item in zip(samples, batch["uav_payloads"]):
        payload = BaseDataElement(
            metainfo=dict(geometry_signature=item["geometry_signature"])
        )
        for key, value in item.items():
            if key != "geometry_signature":
                setattr(payload, key, value)
        device = sample.gt_instances.bboxes.device
        sample.uav = payload.to(device)


def tensor_payload(arrays, rgb, signature):
    payload = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
    payload.update(
        rgb=torch.from_numpy(rgb.copy()).float(), geometry_signature=signature
    )
    return payload
