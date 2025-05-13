# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple

import numpy as np

from vlfm.vlm.blip2itm import BLIP2ITMClient


class Frontier:
    """
    前沿点类，表示机器人可以探索的边界位置
    保存位置坐标和与目标描述的相似度分数
    """

    def __init__(self, xyz: np.ndarray, cosine: float):
        """
        初始化前沿点对象

        Args:
            xyz: 前沿点的3D坐标
            cosine: 该位置图像与目标描述的余弦相似度
        """
        self.xyz = xyz        # 前沿点的空间坐标
        self.cosine = cosine  # 与目标文本的相似度分数


class FrontierMap:
    """
    前沿点地图类，管理和更新探索边界点
    用于基于视觉-语言匹配的目标导向导航
    """

    frontiers: List[Frontier] = []  # 存储所有前沿点的列表

    def __init__(self, encoding_type: str = "cosine"):
        """
        初始化前沿点地图

        Args:
            encoding_type: 编码类型，默认使用"cosine"余弦相似度
        """
        # 创建BLIP2图像-文本匹配客户端，用于计算相似度
        self.encoder: BLIP2ITMClient = BLIP2ITMClient()

    def reset(self) -> None:
        """
        重置前沿点地图，清空所有前沿点
        """
        self.frontiers = []

    def update(self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str) -> None:
        """
        Takes in a list of frontier coordinates and the current image observation from
        the robot. Any stored frontiers that are not present in the given list are
        removed. Any frontiers in the given list that are not already stored are added.
        When these frontiers are added, their cosine field is set to the encoding
        of the given image. The image will only be encoded if a new frontier is added.

        更新前沿点地图：
        1. 移除不在当前列表中的旧前沿点
        2. 添加新的前沿点并计算其与目标描述的相似度
        3. 图像编码仅在添加新前沿点时进行，以节省计算资源

        Args:
            frontier_locations (List[np.ndarray]): 当前检测到的前沿点坐标列表
            curr_image (np.ndarray): 机器人当前的视觉观察图像
            text (str): 目标描述文本，用于与图像比较计算相似度
        """
        # 移除不在当前列表中的前沿点，使用np.array_equal进行坐标比较
        self.frontiers = [
            frontier
            for frontier in self.frontiers
            if any(np.array_equal(frontier.xyz, location) for location in frontier_locations)
        ]

        # 添加不在存储列表中的新前沿点，并设置其相似度
        cosine = None  # 延迟计算相似度直到需要时
        for location in frontier_locations:
            # 检查是否为新的前沿点
            if not any(np.array_equal(frontier.xyz, location) for frontier in self.frontiers):
                # 仅在第一次添加新前沿点时计算相似度，避免重复计算
                if cosine is None:
                    cosine = self._encode(curr_image, text)
                # 创建并添加新前沿点
                self.frontiers.append(Frontier(location, cosine))

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        Encodes the given image using the encoding type specified in the constructor.
        使用指定的编码方式对图像进行编码，计算图像和文本之间的相似度

        Args:
            image (np.ndarray): 要编码的图像
            text (str): 目标描述文本

        Returns:
            float: 图像与文本之间的余弦相似度
        """
        # 调用BLIP2模型计算图像与文本的余弦相似度
        return self.encoder.cosine(image, text)

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        Returns the frontier with the highest cosine and the value of that cosine.
        对前沿点按相似度排序，返回排序后的坐标和相似度值

        Returns:
            Tuple[np.ndarray, List[float]]: 排序后的前沿点坐标数组和对应的相似度值列表
        """
        # 使用np.argsort获取排序后的索引
        cosines = [f.cosine for f in self.frontiers]  # 提取所有相似度
        waypoints = [f.xyz for f in self.frontiers]   # 提取所有坐标
        sorted_inds = np.argsort([-c for c in cosines])  # 降序排序（取负值后升序排序）
        # 根据排序索引重排前沿点
        sorted_values = [cosines[i] for i in sorted_inds]  # 排序后的相似度值
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])  # 排序后的坐标

        return sorted_frontiers, sorted_values
