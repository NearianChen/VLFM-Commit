# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

"""
-----------------------------------好像没用到------------------------------------
"""
# 定义默认的FIBER配置文件路径
DEFAULT_CONFIG = "FIBER/fine_grained/configs/refcocog.yaml"
# 定义默认的预训练模型权重路径
DEFAULT_WEIGHTS = "FIBER/fine_grained/models/fiber_refcocog.pth"


class FIBER:
    """
    FIBER (Fine-grained Instance-level Binding and Expression Recognition)模型类
    用于基于自然语言表达式的细粒度目标检测和识别
    """
    def __init__(self, config_file: str = DEFAULT_CONFIG, weights: str = DEFAULT_WEIGHTS):
        """
        初始化FIBER模型
        
        Args:
            config_file: 模型配置文件路径，默认使用refcocog配置
            weights: 预训练权重文件路径
        """
        # 从配置文件加载配置
        cfg.merge_from_file(config_file)
        # 设置GPU数量为1
        cfg.num_gpus = 1
        # 设置每批次训练图像数量为1
        cfg.SOLVER.IMS_PER_BATCH = 1
        # 设置每批次测试图像数量为1
        cfg.TEST.IMS_PER_BATCH = 1
        # 禁用MDETR风格的类别聚合
        cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
        # 设置评估任务为"grounding"（物体定位）
        cfg.TEST.EVAL_TASK = "grounding"
        # 设置ATSS (Adaptive Training Sample Selection)算法前NMS保留的候选框数量
        cfg.MODEL.ATSS.PRE_NMS_TOP_N = 3000
        # 设置每张图像检测的最大目标数量
        cfg.MODEL.ATSS.DETECTIONS_PER_IMG = 100
        # 设置推理阈值为0，表示不过滤任何检测结果，后续会根据置信度进行过滤
        cfg.MODEL.ATSS.INFERENCE_TH = 0.0
        # 设置模型权重文件路径
        cfg.MODEL.WEIGHT = weights

        # 冻结配置，防止后续修改
        cfg.freeze()

        # 创建FIBER演示对象，设置置信度阈值为0.2
        self.fiber = GLIPDemo(cfg, confidence_threshold=0.2)

    def detect(self, image: np.ndarray, phrase: str, visualize: bool = False) -> ObjectDetections:
        """
        给定图像和描述短语，检测图像中与短语描述最匹配的对象。
        输出对象的边界框被归一化到0和1之间。

        坐标以"xyxy"格式提供（左上角 = x0, y0 和右下角 = x1, y1）。

        Arguments:
            image (np.ndarray): 输入图像，用于检测对象
            phrase (str): 描述要检测对象的短语或表达式
            visualize (bool, optional): 如果为True，将在图像上可视化检测结果
                默认为False

        Returns:
            ObjectDetections: 包含检测结果的数据结构，
                包括源图像、归一化的检测对象边界框、预测分数（logits）、
                用于检测的短语，以及指示是否启用可视化的标志
        """
        # 使用FIBER模型进行推理
        result = self.fiber.inference(image, phrase)
        # 将边界框坐标归一化到0-1范围内（除以图像宽度和高度）
        normalized_bbox = result.bbox / torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        # 创建ObjectDetections对象来存储检测结果
        dets = ObjectDetections(
            image_source=image,  # 源图像
            boxes=normalized_bbox,  # 归一化后的边界框
            logits=result.extra_fields["scores"],  # 预测分数
            phrases=[phrase],  # 使用的描述短语
            fmt="xyxy",  # 坐标格式为xyxy
        )

        return dets


class FIBERClient:
    """
    FIBER模型的客户端类
    通过HTTP请求与服务器端的模型通信
    """
    def __init__(self, url: str = "http://localhost:9080/fiber"):
        """
        初始化FIBER客户端
        
        Args:
            url: FIBER服务的URL地址，默认为本地9080端口
        """
        self.url = url

    def detect(self, image: np.ndarray, phrase: str, visualize: bool = False) -> ObjectDetections:
        """
        通过API调用服务器上的FIBER模型进行对象检测
        
        Args:
            image: 输入图像的numpy数组
            phrase: 描述要检测对象的短语
            visualize: 是否可视化结果，默认为False
            
        Returns:
            包含检测结果的ObjectDetections对象
        """
        # 发送图像和短语到服务器并获取响应
        response = send_request(self.url, image=image, phrase=phrase)["response"]
        # 从JSON响应创建ObjectDetections对象
        detections = ObjectDetections.from_json(response, image_source=image)

        return detections


if __name__ == "__main__":
    """
    主程序入口点，用于启动FIBER服务器
    """
    import argparse  # 命令行参数解析

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9080)  # 添加端口参数，默认9080
    args = parser.parse_args()  # 解析命令行参数

    print("Loading model...")  # 打印加载提示

    class FIBERServer(ServerMixin, FIBER):
        """
        FIBER模型的服务器实现
        继承ServerMixin和FIBER类
        """
        def process_payload(self, payload: dict) -> dict:
            """
            处理HTTP请求负载
            
            Args:
                payload: 包含图像和描述短语的请求负载
                
            Returns:
                包含检测结果的JSON响应
            """
            # 将字符串转换为图像
            image = str_to_image(payload["image"])
            # 执行检测并将结果转换为JSON格式返回
            return {"response": self.detect(image, payload["phrase"]).to_json()}

    # 创建服务器实例
    fiber = FIBERServer()
    print("Model loaded!")  # 打印模型加载完成提示
    print(f"Hosting on port {args.port}...")  # 打印服务器启动信息
    # 启动模型服务器
    host_model(fiber, name="fiber", port=args.port)
