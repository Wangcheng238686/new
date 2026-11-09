#!/usr/bin/env python
"""YOLO11s-seg @ CitySample v1 —— 协议B(匹配增强合同 minimal)训练入口

等效实现(逐参数对齐 Ours):
  - ultralytics 全部内置增强关闭: mosaic/mixup/copy_paste/HSV/flip/尺度/旋转
    /平移/剪切/透视/多尺度 全 0(几何全禁; HSV 由 CSColorJitter 等效替代)
  - patch ultralytics.data.build.build_yolo_dataset: train 模式下在原 transform
    链前置光度增强(fork 安全——patch 发生在 dataloader 构建之前):
      CSColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02,
                    RandomApply p=0.50)
      CSGaussianNoise(sigma=0.02, p=0.50)
  - imgsz=512 原生; 其余(100ep/bs16/auto 优化器/seed=0)ultralytics 默认
"""
import random


import cv2
import numpy as np

CJ_RANGE = dict(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
CJ_P, GN_SIGMA, GN_P = 0.50, 0.02, 0.50


def _blend(a, b, alpha):
    return a * alpha + b * (1 - alpha)


def _brightness(img, factor):
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def _contrast(img, factor):
    b = img.mean(axis=(-3, -2), keepdims=True)
    return np.clip(_blend(img.astype(np.float64), b, factor), 0, 255).astype(np.uint8)


def _saturation(img, factor):
    b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    return np.clip(_blend(img.astype(np.float64), b, factor), 0, 255).astype(np.uint8)


def _hue(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(factor * 179)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


FNS = dict(brightness=_brightness, contrast=_contrast,
           saturation=_saturation, hue=_hue)


def _cs_photometric(img):
    if random.random() < CJ_P:
        ops = list(FNS.items())
        random.shuffle(ops)
        for name, fn in ops:
            r = CJ_RANGE[name]
            factor = (1.0 + random.uniform(-r, r) if name != 'hue'
                      else random.uniform(-r, r))
            img = fn(img, factor)
    if random.random() < GN_P:
        img = img.astype(np.float32) + np.random.normal(
            0.0, GN_SIGMA * 255.0, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img



class _CSWrap:
    """train transform 前置光度增强(可 pickle, worker fork/spawn 均安全)."""

    def __init__(self, orig):
        self.orig = orig

    def __call__(self, labels):
        if isinstance(labels.get('img'), np.ndarray):
            labels['img'] = _cs_photometric(labels['img'])
        return self.orig(labels)


def _install_patch():
    import ultralytics.data.build as _build
    if getattr(_build.build_yolo_dataset, '_cs_minimal', False):
        return
    _orig = _build.build_yolo_dataset

    def patched(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
        ds = _orig(cfg, img_path, batch, data, mode, rect, stride)
        if mode == 'train' and hasattr(ds, 'transform') and ds.transform is not None:
            ds.transform = _CSWrap(ds.transform)
        return ds

    patched._cs_minimal = True
    _build.build_yolo_dataset = patched


def main():
    import os
    import ultralytics
    from ultralytics import YOLO
    _install_patch()
    epochs = int(os.environ.get('CS_EPOCHS', 100))
    batch = int(os.environ.get('CS_BATCH', 16))
    name = os.environ.get('CS_NAME', 'yolo11s_seg_citysample_v1_minimal')
    print('[cs-minimal] patch installed, ultralytics', ultralytics.__version__)
    m = YOLO('/home/wangcheng/project/new/yolo11s-seg.pt')
    data_yaml = ('/data1/wangcheng/dataset/'
                 'CitySample_SatUAV_TrainMinimal_v1_20261021/yolo_seg/data.yaml')
    m.train(
        data=data_yaml, device=0,
        project='/data1/wangcheng/checkpoint/citysample_sat_baselines/yolo11seg',
        name=name, seed=0, exist_ok=True,
        verbose=True, imgsz=512, epochs=epochs, batch=batch,
        # —— 协议B: 内置增强全禁(光度由 patch 等效注入) ——
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, bgr=0.0, multi_scale=False, erasing=0.0)


if __name__ == '__main__':
    main()
