# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    
    生成两种地图：一种表示机器人已经探索过的区域，
    另一种表示机器人已观察到的障碍物。
    """

    _map_dtype: np.dtype = np.dtype(bool)  # 地图数据类型为布尔值
    _frontiers_px: np.ndarray = np.array([])  # 边界点的像素坐标
    frontiers: np.ndarray = np.array([])  # 边界点的世界坐标
    radius_padding_color: tuple = (100, 100, 100)  # 机器人半径填充区域的颜色（灰色）

    def __init__(
        self,
        min_height: float,  # 考虑为障碍物的最小高度
        max_height: float,  # 考虑为障碍物的最大高度
        agent_radius: float,  # 机器人半径，用于计算可通行区域
        area_thresh: float = 3.0,  # 最小边界区域阈值（平方米）
        hole_area_thresh: int = 100000,  # 填充深度图中小洞的面积阈值（像素）
        size: int = 1000,  # 地图尺寸（像素）
        pixels_per_meter: int = 20,  # 每米对应的像素数
    ):
        """
        初始化障碍物地图
        
        Args:
            min_height: 障碍物的最小高度，低于此高度的点不视为障碍
            max_height: 障碍物的最大高度，高于此高度的点不视为障碍
            agent_radius: 机器人的半径，用于计算膨胀后的不可通行区域
            area_thresh: 最小边界区域阈值，小于此值的区域不被视为有效边界
            hole_area_thresh: 深度图中小洞的面积阈值，小于此值的洞将被填充
            size: 地图尺寸（像素）
            pixels_per_meter: 每米对应的像素数，决定地图分辨率
        """
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)  # 已探索区域地图
        self._map = np.zeros((size, size), dtype=bool)  # 障碍物地图
        self._navigable_map = np.zeros((size, size), dtype=bool)  # 可导航区域地图
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)  # 将面积阈值转换为像素
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2  # 计算膨胀核大小，基于机器人半径
        # 将核大小四舍五入到最近的奇数
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建用于膨胀的核

    def reset(self) -> None:
        """
        重置地图，清空所有存储的数据
        """
        super().reset()  # 调用父类的重置方法
        self._navigable_map.fill(0)  # 清空可导航地图
        self.explored_area.fill(0)   # 清空已探索区域
        self._frontiers_px = np.array([])  # 清空边界点像素坐标
        self.frontiers = np.array([])      # 清空边界点世界坐标

    def update_map(
        self,
        depth: Union[np.ndarray, Any],  # 深度图像
        tf_camera_to_episodic: np.ndarray,  # 相机到全局坐标系的变换矩阵
        min_depth: float,  # 深度图的最小值（米）
        max_depth: float,  # 深度图的最大值（米）
        fx: float,  # 相机x方向焦距
        fy: float,  # 相机y方向焦距
        topdown_fov: float,  # 深度相机在俯视图上的视场角
        explore: bool = True,  # 是否更新已探索区域
        update_obstacles: bool = True,  # 是否更新障碍物地图
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.
        
        将当前视图中的所有障碍物添加到地图中，并更新机器人已探索的区域
        
        Args:
            depth: 用于更新障碍物地图的深度图像，归一化到[0,1]范围
            tf_camera_to_episodic: 从相机到全局坐标系的变换矩阵
            min_depth: 深度图的最小值（米）
            max_depth: 深度图的最大值（米） 
            fx: 相机x方向焦距
            fy: 相机y方向焦距
            topdown_fov: 深度相机投影到俯视图上的视场角
            explore: 是否更新已探索区域
            update_obstacles: 是否更新障碍物地图
        """
        # 更新障碍物地图
        if update_obstacles:
            # 处理深度图中的空洞
            if self._hole_area_thresh == -1:
                # 如果不需要填充小洞，将所有0值设为1.0（远处）
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                # 填充小面积的洞
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                
            # 将深度图从归一化值缩放回实际距离
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            # 创建掩码，只保留最大深度范围内的点
            mask = scaled_depth < max_depth
            # 从深度图生成点云（相机坐标系）
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            # 将点云转换到全局坐标系
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            # 根据高度过滤点云，只保留在指定高度范围内的点作为障碍物
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # 将障碍物点云投影到俯视图上
            xy_points = obstacle_cloud[:, :2]  # 提取XY坐标
            pixel_points = self._xy_to_px(xy_points)  # 转换为像素坐标
            # 在障碍物地图上标记障碍物位置
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # 更新可导航区域：对障碍物地图进行膨胀（考虑机器人半径），然后取反
            # 这样障碍物及其周围机器人半径范围内的区域都被标记为不可导航
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        # 如果不需要更新探索区域，直接返回
        if not explore:
            return

        # 更新已探索区域
        # 获取机器人当前位置
        agent_xy_location = tf_camera_to_episodic[:2, 3]  # 提取XY坐标
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]  # 转换为像素坐标
        
        # 使用视野投射算法更新探索区域（揭开战争迷雾）
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),  # 可导航地图作为基础
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),  # 初始无迷雾
            current_point=agent_pixel_location[::-1],  # 当前位置（注意坐标顺序反转）
            current_angle=-extract_yaw(tf_camera_to_episodic),  # 当前朝向角度
            fov=np.rad2deg(topdown_fov),  # 视场角（度）
            max_line_len=max_depth * self.pixels_per_meter,  # 最大视线长度（像素）
        )
        
        # 略微膨胀新探索区域，使其更连续
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        # 更新总探索区域
        self.explored_area[new_explored_area > 0] = 1
        # 确保不可导航区域不被标记为已探索
        self.explored_area[self._navigable_map == 0] = 0
        
        # 寻找已探索区域的轮廓
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,  # 只检索最外层轮廓
            cv2.CHAIN_APPROX_SIMPLE,  # 使用简化的轮廓表示
        )
        
        # 如果存在多个轮廓，选择包含或最接近机器人当前位置的轮廓
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                # 计算点到轮廓的距离（负值表示点在轮廓外，正值表示点在轮廓内）
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:  # 如果点在轮廓内
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:  # 否则找最近的轮廓
                    min_dist = abs(dist)
                    best_idx = idx
                    
            # 创建新的探索区域地图，只保留选定的轮廓
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # -1表示填充轮廓
            self.explored_area = new_area.astype(bool)

        # 计算边界点位置
        self._frontiers_px = self._get_frontiers()  # 获取边界点的像素坐标
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])  # 如果没有边界点，设置为空数组
        else:
            # 将边界点从像素坐标转换为世界坐标
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """
        Returns the frontiers of the map.
        
        计算地图的边界点位置
        边界点是已探索区域与未探索可导航区域之间的交界处
        
        Returns:
            边界点的像素坐标数组
        """
        # 略微膨胀已探索区域，防止已探索区域和不可导航区域之间的小缝隙被误判为边界
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        # 使用边界检测算法寻找边界点
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),  # 可导航区域地图
            explored_area,  # 已探索区域地图
            self._area_thresh_in_pixels,  # 最小边界区域阈值（像素）
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """
        Visualizes the map.
        
        生成地图的可视化图像
        包括已探索区域、障碍物、可导航区域、边界点和机器人轨迹
        
        Returns:
            可视化后的RGB图像
        """
        # 创建白色背景的图像
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # 绘制已探索区域为浅绿色
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # 绘制不可导航区域为灰色
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # 绘制障碍物为黑色
        vis_img[self._map == 1] = (0, 0, 0)
        # 绘制边界点为蓝色
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        # 垂直翻转图像，使Y轴朝上（符合常规坐标系）
        vis_img = cv2.flip(vis_img, 0)

        # 如果有相机位置记录，绘制机器人轨迹
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    """
    根据高度过滤点云数据
    
    Args:
        points: 点云数据数组
        min_height: 最小高度阈值
        max_height: 最大高度阈值
        
    Returns:
        过滤后的点云数据，只包含高度在指定范围内的点
    """
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]
