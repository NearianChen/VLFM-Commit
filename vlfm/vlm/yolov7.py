# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional

import cv2
import numpy as np
import torch

from vlfm.vlm.coco_classes import COCO_CLASSES  # 导入COCO数据集的类别名称
from vlfm.vlm.detections import ObjectDetections  # 导入对象检测结果的数据结构

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image  # 导入服务器相关的工具函数

# 将YOLOv7路径添加到系统路径中，以便导入YOLOv7的模块
sys.path.insert(0, "yolov7/")
try:
    # 尝试导入YOLOv7的相关模块
    from models.experimental import attempt_load  # 用于加载模型
    from utils.datasets import letterbox  # 用于图像预处理，保持纵横比的调整
    from utils.general import (  # 导入通用工具函数
        check_img_size,  # 检查图像尺寸是否合适
        non_max_suppression,  # 非极大值抑制算法
        scale_coords,  # 坐标尺度变换
    )
    from utils.torch_utils import TracedModel  # 用于模型追踪，提高推理速度
except Exception:
    # 如果导入失败，打印提示信息
    print("Could not import yolov7. This is OK if you are only using the client.")
sys.path.pop(0)  # 恢复系统路径


class YOLOv7:
    def __init__(self, weights: str, image_size: int = 640, half_precision: bool = True):
        """
        初始化YOLOv7模型

        Args:
            weights (str): 模型权重文件路径
            image_size (int): 输入图像大小
            half_precision (bool): 是否使用半精度(FP16)计算
        """
        # 检测并设置设备(GPU或CPU)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 如果使用GPU且half_precision为True，则使用半精度
        self.half_precision = self.device.type != "cpu" and half_precision
        # 加载模型权重
        self.model = attempt_load(weights, map_location=self.device)  # 加载FP32模型
        stride = int(self.model.stride.max())  # 获取模型的步长
        self.image_size = check_img_size(image_size, s=stride)  # 检查并调整输入图像尺寸
        # 转换为追踪模型以提高推理速度
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # 转换为FP16精度

        # 模型预热，提高第一次推理速度
        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, int(self.image_size * 0.7), self.image_size).to(self.device)
            if self.half_precision:
                dummy_img = dummy_img.half()
            # 进行三次推理预热
            for i in range(3):
                self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,  # 置信度阈值
        iou_thres: float = 0.45,   # IOU阈值
        classes: Optional[List[str]] = None,  # 过滤特定类别
        agnostic_nms: bool = False,  # 是否使用类别无关的NMS
    ) -> ObjectDetections:
        """
        对输入图像进行目标检测，输出边界框和类别预测数据

        Args:
            image (np.ndarray): RGB图像的numpy数组表示
            conf_thres (float): 过滤检测结果的置信度阈值
            iou_thres (float): 非极大值抑制的IOU阈值
            classes (list): 需要过滤的类别列表
            agnostic_nms (bool): 是否使用类别无关的非极大值抑制
        """
        orig_shape = image.shape  # 保存原始图像尺寸

        # 图像预处理
        # 调整图像大小，保持特定比例
        img = cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=cv2.INTER_AREA,  # 使用区域插值法
        )
        # 使用letterbox调整图像尺寸，保持纵横比
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)  # 转换通道顺序，从HWC变为CHW格式
        img = np.ascontiguousarray(img)  # 确保内存连续，提高运行效率

        # 将numpy数组转换为torch张量并移至目标设备
        img = torch.from_numpy(img).to(self.device)
        # 根据设置转换为半精度或保持浮点精度
        img = img.half() if self.half_precision else img.float()  # uint8转为fp16/32
        img /= 255.0  # 归一化像素值到0.0-1.0范围
        # 如果是3维张量，添加批次维度
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理过程
        with torch.inference_mode():  # 使用推理模式，避免计算梯度导致GPU内存泄漏
            pred = self.model(img)[0]  # 获取模型输出

        # 应用非极大值抑制(NMS)过滤重叠框
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )[0]  # 只取第一个批次的结果

        # 将检测框坐标从模型输出尺寸缩放回原始图像尺寸
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        # 将坐标归一化到0-1范围
        pred[:, 0] /= orig_shape[1]  # x1 / width
        pred[:, 1] /= orig_shape[0]  # y1 / height
        pred[:, 2] /= orig_shape[1]  # x2 / width
        pred[:, 3] /= orig_shape[0]  # y2 / height

        # 提取检测结果的不同部分
        boxes = pred[:, :4]  # 边界框坐标 (x1, y1, x2, y2)
        logits = pred[:, 4]  # 置信度分数
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]  # 将类别索引转换为类别名称

        # 创建检测结果对象
        detections = ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")

        return detections


class YOLOv7Client:
    """
    YOLOv7客户端类，用于与YOLOv7服务器通信
    """
    def __init__(self, port: int = 12184):
        """
        初始化客户端，设置服务器URL

        Args:
            port (int): 服务器端口号
        """
        self.url = f"http://localhost:{port}/yolov7"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        """
        发送图像到服务器并获取检测结果

        Args:
            image_numpy (np.ndarray): 输入图像的numpy数组
        """
        # 发送请求到服务器
        response = send_request(self.url, image=image_numpy)
        # 从JSON响应中恢复检测结果对象
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


# 当直接运行此脚本时执行的代码
if __name__ == "__main__":
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)  # 设置服务器端口参数
    args = parser.parse_args()

    print("Loading model...")

    # 定义YOLOv7服务器类，继承自ServerMixin和YOLOv7
    class YOLOv7Server(ServerMixin, YOLOv7):
        """
        YOLOv7服务器类，处理接收到的请求并返回检测结果
        """
        def process_payload(self, payload: dict) -> dict:
            """处理收到的数据负载，执行图像检测并返回结果"""
            # 将字符串形式的图像转换为numpy数组
            image = str_to_image(payload["image"])
            # 执行预测并转换为JSON格式返回
            return self.predict(image).to_json()

    # 实例化YOLOv7服务器
    yolov7 = YOLOv7Server("data/yolov7-e6e.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    # 启动服务器
    host_model(yolov7, name="yolov7", port=args.port)

    # 以下是测试代码（已注释）
    # # 实例化模型
    # model = YOLOv7(weights="data/yolov7-e6e.pt")
    # img_path = "data/horses.jpg"
    # img = cv2.imread(img_path)
    # # 转换为RGB格式
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 预测
    # pred = model.predict(img)
    # print("Pred")
    # print(pred)
