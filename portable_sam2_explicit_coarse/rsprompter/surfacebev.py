"""Training-free calibrated SurfaceBEV construction used by SURFBEV-P1."""
import torch


def devicez_to_camera_z(devicez_rgb):
    """Decode producer's normalized RGB radix DeviceZ packing into UE cm."""
    x = devicez_rgb.float() / 255.0
    packed = x[..., 0] + x[..., 1] / 255.0 + x[..., 2] / 65025.0
    return 10.0 / (1.0 - packed).clamp_min(1e-8)


@torch.no_grad()
def rasterize_surfacebev(descriptors, devicez_rgb, uav_k, uav_c2w, sat_k, sat_w2c,
                         height_edges, out_hw, sham90=False):
    """Rasterize frozen UAV descriptors into a satellite P3-aligned 12-bin tensor.

    No labels/predictions enter this function.  ``descriptors`` is [V,C,H,W],
    depth is [V,512,512,3], all camera matrices are CV coordinates in UE cm.
    """
    vws, channels, fh, fw = descriptors.shape
    if vws != 5 or devicez_rgb.shape[:3] != (5, 512, 512):
        raise ValueError("SurfaceBEV requires exactly five 512px UAV views")
    oh, ow = out_hw; dev = descriptors.device
    yy, xx = torch.meshgrid(torch.arange(fh, device=dev), torch.arange(fw, device=dev), indexing="ij")
    # Nearest 512-depth sample at each frozen descriptor centre.
    px = ((xx.float() + .5) * 512.0 / fw).long().clamp(max=511)
    py = ((yy.float() + .5) * 512.0 / fh).long().clamp(max=511)
    z = devicez_to_camera_z(devicez_rgb.to(dev))[:, py, px].reshape(vws, -1)
    u = ((xx.reshape(-1).float() + .5) * 512.0 / fw)[None].expand(vws, -1)
    w = ((yy.reshape(-1).float() + .5) * 512.0 / fh)[None].expand(vws, -1)
    pc = torch.stack(((u-uav_k[:,None,0,2])*z/uav_k[:,None,0,0], (w-uav_k[:,None,1,2])*z/uav_k[:,None,1,1], z), -1)
    world = pc @ uav_c2w[:, :3, :3].transpose(1, 2) + uav_c2w[:, None, :3, 3]
    if sham90:
        center = torch.linalg.inv(sat_w2c)[:3, 3]; rel = world[..., :2] - center[:2]
        world[..., 0] = center[0] - rel[..., 1]; world[..., 1] = center[1] + rel[..., 0]
    cam = world @ sat_w2c[:3, :3].T + sat_w2c[:3, 3]
    su = sat_k[0,0]*cam[...,0]/cam[...,2].clamp_min(1e-6)+sat_k[0,2]
    sv = sat_k[1,1]*cam[...,1]/cam[...,2].clamp_min(1e-6)+sat_k[1,2]
    ix, iy = (su / 512.0 * ow).long(), (sv / 512.0 * oh).long()
    valid = (cam[...,2] > 0) & (ix >= 0) & (ix < ow) & (iy >= 0) & (iy < oh)
    flat = (iy.clamp(0, oh-1) * ow + ix.clamp(0, ow-1)).reshape(-1); zsat = cam[...,2].reshape(-1)
    keep_mask = valid.reshape(-1); zmin = torch.full((oh*ow,), float("inf"), device=dev)
    zmin.scatter_reduce_(0, flat[keep_mask], zsat[keep_mask], reduce="amin", include_self=True)
    keep = keep_mask & (zsat <= zmin[flat] + 1e-3)
    feat = descriptors.permute(0,2,3,1).reshape(-1,channels)[keep]; wp = world.reshape(-1,3)[keep]; cell = flat[keep]
    bins = (torch.bucketize(wp[:,2], height_edges.to(dev)) - 1).clamp(0, len(height_edges)-2)
    out = torch.zeros((len(height_edges)-1, channels+5, oh*ow), device=dev); count = torch.zeros((len(height_edges)-1, oh*ow), device=dev)
    nbins = len(height_edges) - 1
    key = bins*oh*ow + cell
    sums = torch.zeros((nbins * oh * ow, channels+2), device=dev)
    sums.index_add_(0, key, torch.cat((feat, wp[:,2:3], wp[:,2:3]**2), 1))
    count.flatten().index_add_(0, key, torch.ones_like(key, dtype=torch.float))
    den = count.clamp_min(1).reshape(-1); packed = sums / den[:,None]
    out[:, :channels+2] = packed.reshape(len(height_edges)-1, oh*ow, channels+2).permute(0,2,1)
    out[:, channels+2] = count.reshape(len(height_edges)-1, oh*ow)
    out[:, channels+3] = (count > 0).reshape(len(height_edges)-1, oh*ow)
    out[:, channels+4] = torch.isfinite(zmin).reshape(oh,ow).flatten()[None]
    return out.reshape(len(height_edges)-1, channels+5, oh, ow), {"input_points": int(vws*fh*fw), "zbuffer_points": int(keep.sum()), "covered_cells": int((count.sum(0)>0).sum())}
