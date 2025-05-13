# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List

import numpy as np

from vlfm.mapping.traj_visualizer import TrajectoryVisualizer


class BaseMap:
    """
    地图的基类，提供基本的地图功能和坐标转换
    用于存储和处理2D栅格地图数据
    """
    _camera_positions: List[np.ndarray] = []  # 相机/机器人位置历史记录
    _last_camera_yaw: float = 0.0            # 相机/机器人最后的航向角
    _map_dtype: np.dtype = np.dtype(np.float32)  # 地图数据类型，默认为32位浮点数

    def __init__(self, size: int = 1000, pixels_per_meter: int = 20, *args: Any, **kwargs: Any):
        """
        初始化地图
        
        Args:
            size: 地图的大小（像素）
            pixels_per_meter: 每米对应的像素数，决定地图的分辨率
        """
        self.pixels_per_meter = pixels_per_meter  # 地图分辨率，每米包含的像素数
        self.size = size  # 地图尺寸（像素）
        self._map = np.zeros((size, size), dtype=self._map_dtype)  # 初始化空白地图
        self._episode_pixel_origin = np.array([size // 2, size // 2])  # 地图中心作为坐标原点
        # 创建轨迹可视化器，用于绘制机器人轨迹
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)

    def reset(self) -> None:
        """
        重置地图数据
        清空地图内容和相机位置历史
        """
        self._map.fill(0)  # 将地图所有单元格设为0
        self._camera_positions = []  # 清空相机位置历史
        # 重新创建轨迹可视化器
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        """
        更新机器人轨迹
        
        Args:
            robot_xy: 机器人在世界坐标系中的(x,y)位置
            robot_heading: 机器人的航向角（弧度）
        """
        self._camera_positions.append(robot_xy)  # 记录当前位置
        self._last_camera_yaw = robot_heading  # 更新航向角

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.
        将世界坐标系中的(x,y)坐标数组转换为像素坐标

        Args:
            points: The array of (x, y) coordinates to convert.
                   要转换的(x,y)坐标数组

        Returns:
            The array of (x, y) pixel coordinates.
            转换后的像素坐标数组
        """
        # 将坐标转换顺序并按分辨率缩放，然后加上原点偏移
        # ::-1反转坐标顺序(x,y)->(y,x)，符合图像坐标系
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        # 沿y轴翻转坐标，因为图像原点在左上角
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)  # 返回整数像素坐标

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.
        将像素坐标数组转换为世界坐标系中的(x,y)坐标

        Args:
            px: The array of pixel coordinates to convert.
               要转换的像素坐标数组

        Returns:
            The array of (x, y) coordinates.
            转换后的世界坐标数组
        """
        # 创建像素坐标的副本，避免修改原数据
        px_copy = px.copy()
        # 沿y轴翻转坐标，恢复正常的坐标系
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        # 减去原点偏移，然后按分辨率缩放
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        # 反转坐标顺序(y,x)->(x,y)，返回世界坐标
        return points[:, ::-1]
