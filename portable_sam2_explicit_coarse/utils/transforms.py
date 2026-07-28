"""
自适应图像和标签缩放Transform
根据输入分辨率自动决定是否上采样
- 512×512 → 上采样到 1024×1024
- 1024×1024 → 保持不变
"""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class AdaptiveResize(BaseTransform):
    """
    自适应Resize Transform，根据输入图像大小自动决定目标分辨率。
    
    如果输入是512×512，则上采样到1024×1024以充分利用SAM2预训练权重。
    如果输入是1024×1024或更高，则保持不变。
    
    Args:
        target_size (int): 小分辨率图像的目标上采样尺寸，默认1024
        keep_ratio (bool): 是否保持宽高比，默认False（强制缩放）
        interpolation (str): 插值方法，默认'bilinear'
    """
    
    rule_map = dict(
        bilinear=cv2.INTER_LINEAR,
        bicubic=cv2.INTER_CUBIC,
        nearest=cv2.INTER_NEAREST,
    )
    
    def __init__(self, target_size=1024, keep_ratio=False, interpolation='bilinear'):
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interpolation = self.rule_map.get(interpolation, cv2.INTER_LINEAR)
    
    def transform(self, results):
        """
        根据输入图像大小自动决定是否上采样。
        
        Args:
            results (dict): 包含 'img', 'gt_bboxes', 'gt_masks' 等的结果字典
            
        Returns:
            dict: 处理后的结果字典，包含:
                - img: 处理后的图像
                - img_shape: 新图像形状
                - ori_shape: 原始图像形状（如果首次处理）
                - scale_factor: 缩放因子
                - gt_bboxes: 缩放后的边界框
                - gt_masks: 缩放后的掩码
        """
        img = results['img']
        h, w = img.shape[:2]
        
        # 获取原始形状（如果之前没有记录）
        if 'ori_shape' not in results:
            results['ori_shape'] = (h, w)
        
        # 判断是否需要上采样
        # 如果图像是512×512（允许1-2像素偏差），则上采样到1024
        # 否则保持原尺寸
        if abs(h - 512) <= 2 and abs(w - 512) <= 2:
            target_h, target_w = self.target_size, self.target_size
            scale_factor_h = self.target_size / h
            scale_factor_w = self.target_size / w
        else:
            # 原生就是大分辨率或其他尺寸，保持不变
            target_h, target_w = h, w
            scale_factor_h = 1.0
            scale_factor_w = 1.0
        
        # 上采样图像
        if (target_h, target_w) != (h, w):
            img = cv2.resize(
                img,
                (target_w, target_h),
                interpolation=self.interpolation
            )
        
        # 更新结果字典
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['scale_factor'] = np.array([scale_factor_w, scale_factor_h, 
                                            scale_factor_w, scale_factor_h],
                                           dtype=np.float32)
        
        # 缩放边界框
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            gt_bboxes = results['gt_bboxes']
            gt_bboxes[:, 0::2] *= scale_factor_w  # x坐标
            gt_bboxes[:, 1::2] *= scale_factor_h  # y坐标
            results['gt_bboxes'] = gt_bboxes
        
        # 缩放掩码
        if 'gt_masks' in results and len(results['gt_masks']) > 0:
            gt_masks = results['gt_masks']
            if hasattr(gt_masks, 'masks'):  # BitmapMasks
                masks_array = gt_masks.masks
            else:
                masks_array = gt_masks
            
            if (target_h, target_w) != (h, w):
                resized_masks = []
                for mask in masks_array:
                    resized_mask = cv2.resize(
                        mask.astype(np.uint8),
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_masks.append(resized_mask)
                
                resized_masks = np.stack(resized_masks, axis=0)
                
                if hasattr(gt_masks, 'masks'):
                    gt_masks.masks = resized_masks
                else:
                    results['gt_masks'] = resized_masks
        
        return results
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'target_size={self.target_size}, '
                f'keep_ratio={self.keep_ratio})')
