# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model.
    
    BLIP 2 图像-文本匹配模型。
    用于评估图像和文本之间的相似度。
    """

    def __init__(
        self,
        name: str = "blip2_image_text_matching",  # 模型名称
        model_type: str = "pretrain",             # 模型类型
        device: Optional[Any] = None,             # 运行设备
    ) -> None:
        """
        初始化BLIP2图像-文本匹配模型
        
        Args:
            name: 模型名称，默认为"blip2_image_text_matching"
            model_type: 模型类型，默认为"pretrain"
            device: 运行设备，如果未指定则优先使用CUDA，否则使用CPU
        """
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # 加载模型和预处理器
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,           # 模型名称
            model_type=model_type,  # 模型类型
            is_eval=True,        # 设置为评估模式
            device=device,       # 运行设备
        )
        self.device = device

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        计算图像和文本之间的余弦相似度。
        
        Args:
            image (numpy.ndarray): 输入图像的numpy数组。
            txt (str): 用于与图像比较的文本。

        Returns:
            float: 图像和文本之间的余弦相似度。值越高表示匹配度越高。
        """
        # 将NumPy数组转换为PIL图像
        pil_img = Image.fromarray(image)
        # 使用视觉预处理器处理图像，并移至指定设备
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        # 使用文本预处理器处理文本
        txt = self.text_processors["eval"](txt)
        # 使用推理模式计算余弦相似度
        with torch.inference_mode():
            # 使用"itc"(image-text contrastive)匹配头计算相似度
            cosine = self.model({"image": img, "text_input": txt}, match_head="itc").item()

        return cosine


class BLIP2ITMClient:
    """
    BLIP2图像-文本匹配模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """
    def __init__(self, port: int = 12182):
        """
        初始化客户端
        
        Args:
            port: 服务器端口号，默认为12182
        """
        self.url = f"http://localhost:{port}/blip2itm"  # 构建API端点URL

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        通过API调用服务器上的模型计算图像和文本的余弦相似度
        
        Args:
            image: 输入图像的numpy数组
            txt: 要比较的文本
            
        Returns:
            图像和文本之间的余弦相似度
        """
        # 打印调试信息，显示图像形状和文本内容
        print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        # 发送请求到服务器并获取响应
        response = send_request(self.url, image=image, txt=txt)
        # 从响应中获取余弦相似度结果并转换为浮点数
        return float(response["response"])


if __name__ == "__main__":
    """
    主程序入口点，用于启动BLIP2ITM服务器
    """
    import argparse  # 命令行参数解析

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)  # 添加端口参数
    args = parser.parse_args()  # 解析命令行参数

    print("Loading model...")  # 打印加载提示

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        """
        BLIP2ITM模型的服务器实现
        继承ServerMixin和BLIP2ITM类
        """
        def process_payload(self, payload: dict) -> dict:
            """
            处理HTTP请求负载
            
            Args:
                payload: 包含图像和文本的请求负载
                
            Returns:
                包含余弦相似度计算结果的响应字典
            """
            # 将字符串转换为图像
            image = str_to_image(payload["image"])
            # 计算余弦相似度并返回结果
            return {"response": self.cosine(image, payload["txt"])}

    # 创建服务器实例
    blip = BLIP2ITMServer()
    print("Model loaded!")  # 打印模型加载完成提示
    print(f"Hosting on port {args.port}...")  # 打印服务器启动信息
    # 启动模型服务器
    host_model(blip, name="blip2itm", port=args.port)
