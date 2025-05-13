# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

# 尝试导入groundingdino模块，如果不可用则打印提示信息
try:
    from groundingdino.util.inference import load_model, predict
except ModuleNotFoundError:
    print("Could not import groundingdino. This is OK if you are only using the client.")

# 定义Grounding DINO模型的配置文件路径
GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# 定义预训练权重文件路径
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
# 定义默认检测类别，使用特殊格式：类别之间用 " . " 分隔，最后以 " ." 结尾
CLASSES = "chair . person . dog ."  # 默认类别，可在推理时覆盖


class GroundingDINO:
    """
    Grounding DINO (Detection with Input and Output)模型类
    用于基于文本提示的对象检测
    """
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,  # 模型配置文件路径
        weights_path: str = GROUNDING_DINO_WEIGHTS,  # 模型权重文件路径
        caption: str = CLASSES,  # 默认检测类别
        box_threshold: float = 0.35,  # 边界框置信度阈值
        text_threshold: float = 0.25,  # 文本匹配置信度阈值
        device: torch.device = torch.device("cuda"),  # 运行设备
    ):
        """
        初始化Grounding DINO模型
        
        Args:
            config_path: 模型配置文件路径
            weights_path: 预训练权重文件路径
            caption: 默认检测类别字符串
            box_threshold: 边界框检测置信度阈值
            text_threshold: 文本匹配置信度阈值
            device: 运行设备，默认为CUDA
        """
        # 加载模型并移至指定设备
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path).to(device)
        self.caption = caption  # 保存默认类别
        self.box_threshold = box_threshold  # 保存边界框阈值
        self.text_threshold = text_threshold  # 保存文本阈值

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        """
        对输入图像进行对象检测预测
        
        Arguments:
            image (np.ndarray): 输入图像，以numpy数组形式
            caption (Optional[str]): 包含可能类别的字符串，用句点分隔。
                                     如果未提供，将使用默认类别
        
        Returns:
            ObjectDetections: 包含检测结果的ObjectDetections实例
        """
        # 将图像转换为tensor并归一化(0-255到0-1)
        image_tensor = F.to_tensor(image)
        # 使用ImageNet均值和标准差进行标准化
        image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 确定使用的类别提示
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)  # 打印使用的类别提示
        # 使用推理模式，禁用梯度计算
        with torch.inference_mode():
            # 调用模型进行预测
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,  # 边界框置信度阈值
                text_threshold=self.text_threshold,  # 文本匹配置信度阈值
            )
        # 创建检测结果对象
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # 移除类名不完全匹配提供类别的检测结果
        classes = caption_to_use[: -len(" .")].split(" . ")  # 解析类别列表
        detections.filter_by_class(classes)  # 根据类别列表过滤检测结果

        return detections


class GroundingDINOClient:
    """
    Grounding DINO模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """
    def __init__(self, port: int = 12181):
        """
        初始化客户端
        
        Args:
            port: 服务器端口号，默认为12181
        """
        self.url = f"http://localhost:{port}/gdino"  # 构建API端点URL

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        """
        通过API调用服务器上的模型进行预测
        
        Args:
            image_numpy: 输入图像的numpy数组
            caption: 类别提示字符串，可选
            
        Returns:
            包含检测结果的ObjectDetections对象
        """
        # 发送请求到服务器
        response = send_request(self.url, image=image_numpy, caption=caption)
        # 从JSON响应创建ObjectDetections对象
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    """
    主程序入口点，用于启动Grounding DINO服务器
    """
    import argparse  # 命令行参数解析

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)  # 添加端口参数
    args = parser.parse_args()  # 解析命令行参数

    print("Loading model...")  # 打印加载提示

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        """
        Grounding DINO模型的服务器实现
        继承ServerMixin和GroundingDINO类
        """
        def process_payload(self, payload: dict) -> dict:
            """
            处理HTTP请求负载
            
            Args:
                payload: 包含图像和文本提示的请求负载
                
            Returns:
                JSON格式的检测结果
            """
            # 将字符串转换为图像
            image = str_to_image(payload["image"])
            # 执行预测并转换为JSON格式返回
            return self.predict(image, caption=payload["caption"]).to_json()

    # 创建服务器实例
    gdino = GroundingDINOServer()
    print("Model loaded!")  # 打印模型加载完成提示
    print(f"Hosting on port {args.port}...")  # 打印服务器启动信息
    # 启动模型服务器
    host_model(gdino, name="gdino", port=args.port)
