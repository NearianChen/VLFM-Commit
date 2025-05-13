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


class BLIP2:
    """
    BLIP2模型类
    用于图像理解、描述生成和视觉问答任务
    BLIP: Bootstrapping Language-Image Pre-training
    """
    def __init__(
        self,
        name: str = "blip2_t5",           # 模型名称，默认使用T5变体
        model_type: str = "pretrain_flant5xxl",  # 模型类型，默认使用预训练的flant5xxl
        device: Optional[Any] = None,     # 运行设备
    ) -> None:
        """
        初始化BLIP2模型
        
        Args:
            name: 模型名称，决定使用哪种BLIP2变体
            model_type: 模型类型，指定预训练模型的大小和架构
            device: 运行设备，如果未指定则优先使用CUDA，否则使用CPU
        """
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # 加载模型和预处理器，使用lavis库提供的函数
        # 第三个返回值是文本处理器，这里使用_忽略它
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name,           # 模型名称
            model_type=model_type,  # 模型类型
            is_eval=True,        # 设置为评估模式
            device=device,       # 运行设备
        )
        self.device = device

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """Generates a caption for the given image.
        为给定图像生成描述或回答关于图像的问题。

        Args:
            image (numpy.ndarray): The input image as a numpy array.
                                  输入图像的numpy数组
            prompt (str, optional): An optional prompt to provide context and guide
                the caption generation. Can be used to ask questions about the image.
                可选的提示文本，用于提供上下文并引导描述生成。
                可以用来询问关于图像的问题。

        Returns:
            dict: The generated caption.
                 生成的描述文本或问题的回答。
        """
        # 将NumPy数组转换为PIL图像
        pil_img = Image.fromarray(image)
        # 使用推理模式，避免计算梯度
        with torch.inference_mode():
            # 处理图像并移至指定设备
            processed_image = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
            # 根据是否提供prompt决定生成方式
            if prompt is None or prompt == "":
                # 如果没有提示，则生成图像描述
                out = self.model.generate({"image": processed_image})[0]
            else:
                # 如果有提示，则根据提示和图像生成回答
                out = self.model.generate({"image": processed_image, "prompt": prompt})[0]

        return out


class BLIP2Client:
    """
    BLIP2模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """
    def __init__(self, port: int = 12185):
        """
        初始化客户端
        
        Args:
            port: 服务器端口号，默认为12185
        """
        self.url = f"http://localhost:{port}/blip2"  # 构建API端点URL

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        通过API调用服务器上的BLIP2模型获取图像描述或问答
        
        Args:
            image: 输入图像的numpy数组
            prompt: 可选的提示文本，用于引导描述或提问
            
        Returns:
            生成的描述或问题回答
        """
        # 如果prompt为None，则将其设为空字符串
        if prompt is None:
            prompt = ""
        # 发送请求到服务器
        response = send_request(self.url, image=image, prompt=prompt)
        # 返回响应结果
        return response["response"]


if __name__ == "__main__":
    """
    主程序入口点，用于启动BLIP2服务器
    """
    import argparse  # 命令行参数解析

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)  # 添加端口参数
    args = parser.parse_args()  # 解析命令行参数

    print("Loading model...")  # 打印加载提示

    class BLIP2Server(ServerMixin, BLIP2):
        """
        BLIP2模型的服务器实现
        继承ServerMixin和BLIP2类
        """
        def process_payload(self, payload: dict) -> dict:
            """
            处理HTTP请求负载
            
            Args:
                payload: 包含图像和可选提示的请求负载
                
            Returns:
                包含生成结果的响应字典
            """
            # 将字符串转换为图像
            image = str_to_image(payload["image"])
            # 调用ask方法并返回结果
            return {"response": self.ask(image, payload.get("prompt"))}

    # 使用较小的flant5xl模型而非注释中的opt2.7b模型
    # blip = BLIP2Server(name="blip2_opt", model_type="pretrain_opt2.7b")
    blip = BLIP2Server(name="blip2_t5", model_type="pretrain_flant5xl")
    print("Model loaded!")  # 打印模型加载完成提示
    print(f"Hosting on port {args.port}...")  # 打印服务器启动信息
    # 启动模型服务器
    host_model(blip, name="blip2", port=args.port)
