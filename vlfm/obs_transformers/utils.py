# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Tuple

import torch
from torch import Tensor


def image_resize(
    img: Tensor,
    size: Tuple[int, int],
    channels_last: bool = False,
    interpolation_mode: str = "area",
) -> torch.Tensor:
    """Resizes an img.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    # 将输入转换为PyTorch张量，确保统一数据类型
    img = torch.as_tensor(img)
    
    # 检查输入张量是否没有批次维度（仅有单张图像）
    no_batch_dim = len(img.shape) == 3
    
    # 检查输入张量维度是否在支持范围内（3-5维）
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    
    # 如果没有批次维度，添加一个批次维度以适应interpolate函数要求
    if no_batch_dim:
        img = img.unsqueeze(0)  # 添加批次维度
    
    # 如果通道在最后一维，需要调整张量维度顺序以符合PyTorch的插值要求
    if channels_last:
        if len(img.shape) == 4:
            # NHWC (批次,高度,宽度,通道) -> NCHW (批次,通道,高度,宽度)
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC (批次,深度,高度,宽度,通道) -> NDCHW (批次,深度,通道,高度,宽度)
            # 用于3D数据，如视频或体积数据
            img = img.permute(0, 1, 4, 2, 3)

    # 使用PyTorch的插值函数调整图像大小
    # 转换为浮点型以提高精度，然后再转换回原始数据类型
    img = torch.nn.functional.interpolate(img.float(), size=size, mode=interpolation_mode).to(dtype=img.dtype)
    
    # 如果原始输入是channels_last格式，需要将结果转换回相应格式
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
            
    # 如果原始输入没有批次维度，移除添加的批次维度，恢复原始维度结构
    if no_batch_dim:
        img = img.squeeze(dim=0)  # 移除批次维度
        
    return img
