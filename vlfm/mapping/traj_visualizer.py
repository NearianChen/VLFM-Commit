# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Union

import cv2
import numpy as np


class TrajectoryVisualizer:
    """
    轨迹可视化工具，用于在地图上绘制机器人的移动路径和当前位置
    """
    _num_drawn_points: int = 1  # 已绘制点的数量，用于增量绘制
    _cached_path_mask: Union[np.ndarray, None] = None  # 缓存的路径掩码，避免重复绘制
    _origin_in_img: Union[np.ndarray, None] = None  # 图像中的原点位置
    _pixels_per_meter: Union[float, None] = None  # 每米对应的像素数（地图比例尺）
    agent_line_length: int = 10  # 表示机器人朝向的线段长度
    agent_line_thickness: int = 3  # 朝向线段的粗细
    path_color: tuple = (0, 255, 0)  # 路径颜色，绿色
    path_thickness: int = 3  # 路径线条粗细
    scale_factor: float = 1.0  # 可视化元素的缩放因子

    def __init__(self, origin_in_img: np.ndarray, pixels_per_meter: float):
        """
        初始化轨迹可视化器

        Args:
            origin_in_img: 图像中的原点位置，通常为地图中心
            pixels_per_meter: 每米对应的像素数，决定可视化比例
        """
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter

    def reset(self) -> None:
        """
        重置可视化器状态，清空缓存和计数器
        """
        self._num_drawn_points = 1  # 重置已绘制点数
        self._cached_path_mask = None  # 清空路径掩码缓存

    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,
    ) -> np.ndarray:
        """Draws the trajectory on the image and returns it

        在图像上绘制完整轨迹，包括历史路径和当前机器人位置

        Args:
            img: 要绘制的基础图像
            camera_positions: 机器人位置历史记录
            camera_yaw: 机器人当前朝向角度

        Returns:
            添加了轨迹的图像
        """
        img = self._draw_path(img, camera_positions)  # 先绘制历史路径
        img = self._draw_agent(img, camera_positions[-1], camera_yaw)  # 再绘制当前机器人位置
        return img

    def _draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Draws the path on the image and returns it

        在图像上绘制移动路径，使用缓存实现增量绘制，提高效率

        Args:
            img: 要绘制的基础图像
            camera_positions: 机器人位置历史记录

        Returns:
            添加了路径的图像
        """
        if len(camera_positions) < 2:
            return img  # 至少需要两个点才能绘制路径

        # 使用缓存的路径掩码或创建新的掩码
        if self._cached_path_mask is not None:
            path_mask = self._cached_path_mask.copy()
        else:
            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 增量绘制，只绘制新增的路径段
        for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
            path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])

        # 将路径叠加到图像上，使用指定的路径颜色
        img[path_mask == 255] = self.path_color

        # 更新缓存和计数器
        self._cached_path_mask = path_mask
        self._num_drawn_points = len(camera_positions)

        return img

    def _draw_line(self, img: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> np.ndarray:
        """Draws a line between two points and returns it

        在图像上绘制两点之间的直线

        Args:
            img: 要绘制的图像（掩码）
            pt_a: 第一个点的世界坐标
            pt_b: 第二个点的世界坐标

        Returns:
            添加了线段的图像
        """
        # 将世界坐标转换为图像像素坐标
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        # 如果两点重合，不需要绘制线段
        if np.array_equal(px_a, px_b):
            return img

        # 绘制线段（注意坐标顺序翻转）
        cv2.line(
            img,
            tuple(px_a[::-1]),  # 翻转坐标顺序以适应OpenCV的xy顺序
            tuple(px_b[::-1]),
            255,  # 在掩码上，线段颜色为255
            int(self.path_thickness * self.scale_factor),  # 应用缩放因子
        )

        return img

    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """Draws the agent on the image and returns it

        在图像上绘制机器人的当前位置和朝向
        机器人由一个圆点和一条表示朝向的线段组成

        Args:
            img: 要绘制的图像
            camera_position: 机器人当前位置的世界坐标
            camera_yaw: 机器人当前朝向角度

        Returns:
            添加了机器人标记的图像
        """
        # 将机器人位置坐标转换为像素坐标
        px_position = self._metric_to_pixel(camera_position)

        # 绘制表示机器人位置的圆点（黄色）
        cv2.circle(
            img,
            tuple(px_position[::-1]),  # 翻转坐标顺序
            int(8 * self.scale_factor),  # 圆点半径，应用缩放因子
            (255, 192, 15),  # 橙黄色
            -1,  # 填充圆
        )

        # 计算朝向线段的终点
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )

        # 绘制表示机器人朝向的线段（黑色）
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),  # 黑色
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img

    def draw_circle(self, img: np.ndarray, position: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Draws the point as a circle on the image and returns it

        在图像上绘制圆形标记点

        Args:
            img: 要绘制的图像
            position: 点的世界坐标
            **kwargs: 传递给cv2.circle的其他参数

        Returns:
            添加了圆形标记的图像
        """
        # 将世界坐标转换为像素坐标
        px_position = self._metric_to_pixel(position)
        # 绘制圆形（注意坐标顺序翻转）
        cv2.circle(img, tuple(px_position[::-1]), **kwargs)

        return img

    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """Converts a metric coordinate to a pixel coordinate

        将世界坐标系中的点转换为图像像素坐标系中的点
        需要考虑坐标系方向、比例尺和原点偏移

        Args:
            pt: 世界坐标系中的点

        Returns:
            对应的图像像素坐标
        """
        # 需要翻转y轴，因为像素坐标从左上角开始
        # 世界坐标乘以比例尺并翻转，然后加上原点偏移
        px = pt * self._pixels_per_meter * np.array([-1, -1]) + self._origin_in_img
        # 下面这行是替代方案，不翻转坐标（已注释掉）
        # px = pt * self._pixels_per_meter + self._origin_in_img
        px = px.astype(np.int32)  # 转换为整数像素坐标
        return px
