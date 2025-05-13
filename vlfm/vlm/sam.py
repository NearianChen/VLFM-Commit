# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, List, Optional

import numpy as np
import torch
import functools
from .server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    send_request,
    str_to_bool_arr,
    str_to_image,
)

# 尝试导入mobile_sam库，如果不可用则打印提示信息
try:
    from mobile_sam import SamPredictor, sam_model_registry
except ModuleNotFoundError:
    print("Could not import mobile_sam. This is OK if you are only using the client.")


class MobileSAM:
    """
    移动版本的Segment Anything Model (SAM)
    用于图像分割任务，特别是针对边界框内的对象
    """
    def __init__(
        self,
        sam_checkpoint: str,  # 模型检查点文件路径
        model_type: str = "vit_t",  # 模型类型，默认为tiny版本的ViT
        device: Optional[Any] = None,  # 运行设备，可选
    ) -> None:
        """
        初始化Mobile SAM模型
        
        Args:
            sam_checkpoint: 模型权重文件路径
            model_type: 模型类型，默认为"vit_t"(tiny ViT)
            device: 运行设备，如果不指定则优先使用CUDA，否则使用CPU
        """
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        # 应用补丁以解决PyTorch 2.6中torch.load的weights_only默认值问题
        original_torch_load = torch.load
        torch.load = functools.partial(original_torch_load, weights_only=False)
        
        try:
            # 从注册表加载指定类型的SAM模型
            mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            mobile_sam.to(device=device)  # 将模型移至指定设备
            mobile_sam.eval()  # 设置为评估模式
            self.predictor = SamPredictor(mobile_sam)  # 创建预测器
        finally:
            # 恢复原始函数
            torch.load = original_torch_load

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        分割图像中指定边界框内的对象。

        Args:
            image (numpy.ndarray): 输入图像的numpy数组
            bbox (List[int]): 边界框，格式为 [x1, y1, x2, y2]

        Returns:
            np.ndarray: 分割结果，布尔类型的掩码数组，大小与输入图像相同
        """
        with torch.inference_mode():  # 推理模式，禁用梯度计算
            self.predictor.set_image(image)  # 设置预测器的输入图像
            masks, _, _ = self.predictor.predict(box=np.array(bbox), multimask_output=False)  # 预测分割掩码

        return masks[0]  # 返回第一个掩码结果


class MobileSAMClient:
    """
    移动版SAM模型的客户端，
    通过HTTP请求与服务器端的模型通信
    """
    def __init__(self, port: int = 12183):
        """
        初始化客户端
        
        Args:
            port: 服务器端口号，默认为12183
        """
        self.url = f"http://localhost:{port}/mobile_sam"  # 构建API端点URL

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        通过API调用服务器上的模型来分割图像中的对象
        
        Args:
            image: 输入图像的numpy数组
            bbox: 边界框，格式为 [x1, y1, x2, y2]
            
        Returns:
            分割结果，布尔类型的掩码数组
        """
        # 发送请求到服务器
        response = send_request(self.url, image=image, bbox=bbox)
        # 从响应中获取编码后的掩码
        cropped_mask_str = response["cropped_mask"]
        # 将字符串转换回布尔数组
        cropped_mask = str_to_bool_arr(cropped_mask_str, shape=tuple(image.shape[:2]))

        return cropped_mask


if __name__ == "__main__":
    """
    主程序入口点，用于启动Mobile SAM服务器
    """
    import argparse  # 命令行参数解析

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)  # 添加端口参数
    args = parser.parse_args()  # 解析命令行参数

    print("Loading model...")

    class MobileSAMServer(ServerMixin, MobileSAM):
        """
        Mobile SAM模型的服务器实现，
        继承ServerMixin和MobileSAM类
        """
        def process_payload(self, payload: dict) -> dict:
            """
            处理HTTP请求负载
            
            Args:
                payload: 包含图像和边界框信息的请求负载
                
            Returns:
                包含分割掩码的响应字典
            """
            # 将字符串转换为图像
            image = str_to_image(payload["image"])
            # 执行分割操作
            cropped_mask = self.segment_bbox(image, payload["bbox"])
            # 将掩码转换为字符串用于传输
            cropped_mask_str = bool_arr_to_str(cropped_mask)
            # 返回结果
            return {"cropped_mask": cropped_mask_str}

    # 创建服务器实例，从环境变量或默认路径加载模型
    mobile_sam = MobileSAMServer(sam_checkpoint=os.environ.get("MOBILE_SAM_CHECKPOINT", "data/mobile_sam.pt"))
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    # 启动模型服务器
    host_model(mobile_sam, name="mobile_sam", port=args.port)
