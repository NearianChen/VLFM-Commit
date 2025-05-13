# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_rotation_matrix
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

# 全局调试和记录设置
DEBUG = False  # 是否开启调试模式
SAVE_VISUALIZATIONS = False  # 是否保存可视化结果
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"  # 是否记录地图数据，通过环境变量控制
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"  # 是否回放记录的地图数据
RECORDING_DIR = "value_map_recordings"  # 记录数据的目录
JSON_PATH = osp.join(RECORDING_DIR, "data.json")  # 记录数据的JSON文件路径
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")  # 记录初始化参数的JSON文件路径


class ValueMap(BaseMap):
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object.
    
    生成一个价值地图，表示环境中已探索区域对于寻找和导航到目标对象的价值程度。
    这是一种用于引导机器人探索的启发式地图。
    """

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}  # 缓存的视场角和深度对应的置信度掩码
    _camera_positions: List[np.ndarray] = []  # 相机历史位置列表
    _last_camera_yaw: float = 0.0  # 相机最后的航向角
    _min_confidence: float = 0.25  # 最小置信度阈值
    _decision_threshold: float = 0.35  # 决策阈值，低于此值的区域被认为不可靠
    _map: np.ndarray  # 主地图数组，表示每个区域的总体置信度

    def __init__(
        self,
        value_channels: int,  # 价值地图的通道数
        size: int = 1000,  # 地图大小（像素）
        use_max_confidence: bool = True,  # 是否使用最大置信度值
        fusion_type: str = "default",  # 融合类型
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        """
        Args:
            value_channels: The number of channels in the value map.
                           价值地图的通道数，每个通道可以表示不同类型的价值
            size: The size of the value map in pixels.
                  地图大小（像素），默认1000x1000
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
                是否使用最大置信度值，否则使用加权平均置信度值
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
                融合新旧数据时使用的策略类型
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
                可选的障碍物地图，用于覆盖视场中被遮挡的区域
        """
        # 如果是回放模式，使用更大的地图尺寸
        if PLAYING:
            size = 2000
        # 调用父类初始化
        super().__init__(size)
        # 初始化价值地图，形状为(size, size, value_channels)
        self._value_map = np.zeros((size, size, value_channels), np.float32)
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map
        # 如果提供了障碍物地图，确保其分辨率和大小与当前地图一致
        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        # 允许通过环境变量覆盖融合类型设置
        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        # 配置记录模式
        if RECORDING:
            # 如果记录目录已存在，则删除它
            if osp.isdir(RECORDING_DIR):
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # 将所有初始化参数保存到文件
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # 创建一个空的JSON文件作为数据存储的初始文件
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        """
        重置价值地图，清空所有存储的数据
        """
        super().reset()  # 调用父类的重置方法
        self._value_map.fill(0)  # 将价值地图所有值设为0

    def update_map(
        self,
        values: np.ndarray,  # 要使用的价值数组
        depth: np.ndarray,  # 深度图像
        tf_camera_to_episodic: np.ndarray,  # 从相机到全局坐标系的变换矩阵
        min_depth: float,  # 最小深度值（米）
        max_depth: float,  # 最大深度值（米）
        fov: float,  # 相机视场角（弧度）
    ) -> None:
        """Updates the value map with the given depth image, pose, and value to use.
        
        使用给定的深度图像、位姿和价值更新价值地图
        
        Args:
            values: 用于更新地图的价值数组
            depth: 用于更新地图的深度图像，预期已归一化到[0,1]范围
            tf_camera_to_episodic: 从相机坐标系到全局坐标系的变换矩阵
            min_depth: 深度图像的最小深度值（米）
            max_depth: 深度图像的最大深度值（米）
            fov: 相机视场角（弧度）
        """
        # 确保提供的价值数组长度与地图通道数匹配
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        # 将新数据定位到全局地图中
        curr_map = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)

        # 融合新数据与现有数据
        self._fuse_new_data(curr_map, values)

        # 如果开启记录模式，保存当前数据
        if RECORDING:
            # 获取下一个图像文件索引
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            # 保存深度图像
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            # 读取已有数据
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            # 添加新数据
            data[img_path] = {
                "values": values.tolist(),
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            # 写回文件
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """Selects the best waypoint from the given list of waypoints.
        
        从给定的路点列表中选择最佳路点。
        根据价值地图中的值对路点进行排序，返回排序后的路点和对应的价值。
        
        Args:
            waypoints: 2D路点数组，包含多个候选路点
            radius: 用于选择最佳路点的半径（米）
            reduce_fn: 用于降维多通道价值的函数，默认为np.max
            
        Returns:
            排序后的路点数组和对应的价值列表
        """
        # 将半径从米转换为像素
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            """
            获取给定点周围区域的价值
            
            Args:
                point: 世界坐标系中的点坐标
                
            Returns:
                对应点周围区域的价值（单值或元组）
            """
            x, y = point
            # 将坐标转换为像素坐标
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            # 获取每个通道的值
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            # 根据通道数返回值或元组
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        # 计算每个路点的价值
        values = [get_value(point) for point in waypoints]

        # 对于多通道情况，使用指定的函数进行降维
        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(values)

        # 使用np.argsort获取排序后的索引（降序）
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,  # 要在地图上绘制的标记点
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),  # 用于降维多通道地图的函数
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """Return an image representation of the map
        
        返回地图的图像表示
        将价值地图转换为彩色图像，便于可视化
        
        Args:
            markers: 要在地图上绘制的标记点列表，每项包含坐标和绘制参数
            reduce_fn: 用于将多通道地图转换为单通道的函数，默认取最大值
            obstacle_map: 可选的障碍物地图，用于遮盖未探索区域
            
        Returns:
            地图的RGB图像表示
        """
        # 使用提供的函数将多通道地图降维为单通道
        reduced_map = reduce_fn(self._value_map).copy()
        # 如果提供了障碍物地图，将未探索区域设为0
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        # 上下翻转地图以匹配正常坐标系方向
        map_img = np.flipud(reduced_map)
        # 将所有0值临时设为最大值，避免影响颜色映射（稍后会恢复）
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        # 将单通道地图转换为"inferno"彩色图像
        map_img = monochannel_to_inferno_rgb(map_img)
        # 恢复所有原始为0的值为白色
        map_img[zero_mask] = (255, 255, 255)
        
        # 如果有相机位置历史，绘制轨迹
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

            # 如果提供了标记点，绘制到图像上
            if markers is not None:
                for pos, marker_kwargs in markers:
                    map_img = self._traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        return map_img

    def _process_local_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.
        
        使用视场角和深度图像，返回FOV内可见部分的掩码。
        通过处理深度图像确定障碍物的位置，并在FOV锥体中剔除被遮挡的区域。
        
        Args:
            depth: 用于确定FOV可见部分的深度图像
            fov: 视场角（弧度）
            min_depth: 最小深度值（米）
            max_depth: 最大深度值（米）
            
        Returns:
            FOV可见部分的掩码
        """
        # 如果深度图是3D数组，压缩通道维度
        if len(depth.shape) == 3:
            depth = depth.squeeze(2)
        # 将深度图压缩为一行，每列取最大深度值
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

        # 创建与深度行等长的角度数组，从-fov/2到fov/2
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

        # 根据角度和深度值为每个点分配x,y坐标
        x = depth_row
        y = depth_row * np.tan(angles)

        # 获取置信度掩码
        cone_mask = self._get_confidence_mask(fov, max_depth)

        # 将x,y坐标转换为像素坐标
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # 从x,y坐标创建轮廓，添加图像左上角和右上角作为起始点
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)

        # 在锥体掩码上绘制轮廓，填充为黑色（表示不可见区域）
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # type: ignore

        if DEBUG:
            vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
            if SAVE_VISUALIZATIONS:
                # Create visualizations directory if it doesn't exist
                if not os.path.exists("visualizations"):
                    os.makedirs("visualizations")
                # Expand the depth_row back into a full image
                depth_row_full = np.repeat(depth_row.reshape(1, -1), depth.shape[0], axis=0)
                # Stack the depth images with the visible mask
                depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                depth_row_full = cv2.cvtColor((depth_row_full * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis = np.flipud(vis)
                new_width = int(vis.shape[1] * (depth_rgb.shape[0] / vis.shape[0]))
                vis_resized = cv2.resize(vis, (new_width, depth_rgb.shape[0]))
                vis = np.hstack((depth_rgb, depth_row_full, vis_resized))
                time_id = int(time.time() * 1000)
                cv2.imwrite(f"visualizations/{time_id}.png", vis)
            else:
                cv2.imshow("obstacle mask", vis)
                cv2.waitKey(0)

        return visible_mask

    def _localize_new_data(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        """
        将局部观察数据定位到全局地图中
        
        Args:
            depth: 深度图像
            tf_camera_to_episodic: 从相机到全局坐标系的变换矩阵
            min_depth: 最小深度值（米）
            max_depth: 最大深度值（米）
            fov: 视场角（弧度）
            
        Returns:
            定位到全局地图坐标系的当前观察数据
        """
        # 获取局部FOV数据
        curr_data = self._process_local_data(depth, fov, min_depth, max_depth)

        # 根据相机朝向旋转数据
        yaw = extract_yaw(tf_camera_to_episodic)
        # 回放模式下的特殊处理
        if PLAYING:
            if yaw > 0:
                yaw = 0
            else:
                yaw = np.deg2rad(30)
        # 旋转图像以匹配相机方向
        curr_data = rotate_image(curr_data, -yaw)

        # 确定掩码应该叠加的位置
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

        # 将坐标转换为像素单位
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # 将新数据叠加到全局地图上
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        return curr_map

    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered
        
        生成一个不考虑障碍物的FOV锥体掩码
        
        Args:
            fov: 视场角（弧度）
            max_depth: 最大深度（米）
            
        Returns:
            表示FOV锥体的二维数组
        """
        # 计算锥体大小（像素）
        size = int(max_depth * self.pixels_per_meter)
        # 创建空白掩码
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        # 使用椭圆绘制FOV锥体
        cone_mask = cv2.ellipse(  # type: ignore
            cone_mask,
            (size, size),  # 中心像素
            (size, size),  # 轴长度
            0,  # 旋转角度
            -np.rad2deg(fov) / 2 + 90,  # 起始角度
            np.rad2deg(fov) / 2 + 90,  # 结束角度
            1,  # 颜色（1表示可见区域）
            -1,  # 厚度（-1表示填充）
        )
        return cone_mask

    def _get_confidence_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily
        
        生成一个FOV锥体掩码，中心区域权重更高
        根据与中心轴的角度调整每个点的置信度值
        
        Args:
            fov: 视场角（弧度）
            max_depth: 最大深度（米）
            
        Returns:
            带权重的FOV锥体掩码
        """
        # 检查缓存中是否已存在此参数组合的掩码
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()
            
        # 获取空白锥体掩码
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        # 创建调整后的掩码
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        
        # 遍历掩码的每个点
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                # 计算点到中心的水平和垂直距离
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                # 计算点到中心轴的角度
                angle = np.arctan2(vertical, horizontal)
                # 将角度从[0,fov/2]重映射到[0,pi/2]
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                # 使用余弦平方计算置信度，中心值更高
                confidence = np.cos(angle) ** 2
                # 重映射置信度值到[min_confidence,1]范围
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
                
        # 将置信度掩码与锥体掩码相乘，使锥体外区域为0
        adjusted_mask = adjusted_mask * cone_mask
        # 缓存计算结果
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()

        return adjusted_mask

    def _fuse_new_data(self, new_map: np.ndarray, values: np.ndarray) -> None:
        """Fuse the new data with the existing value and confidence maps.
        
        将新数据与现有的价值图和置信度图融合
        
        Args:
            new_map: 新的地图数据，置信度在0到1之间，1为最高
            values: 新观测区域的价值
        """
        # 确保提供的价值数量与地图通道数匹配
        assert (
            len(values) == self._value_channels
        ), f"Incorrect number of values given ({len(values)}). Expected {self._value_channels}."

        # 如果提供了障碍物地图，使用它来遮盖未探索区域
        if self._obstacle_map is not None:
            explored_area = self._obstacle_map.explored_area
            new_map[explored_area == 0] = 0
            self._map[explored_area == 0] = 0
            self._value_map[explored_area == 0] *= 0

        # 根据不同的融合类型选择不同的融合策略
        if self._fusion_type == "replace":
            # 替代模式：当前观察的值会覆盖任何现有值
            print("VALUE MAP ABLATION:", self._fusion_type)
            new_value_map = np.zeros_like(self._value_map)
            new_value_map[new_map > 0] = values
            self._map[new_map > 0] = new_map[new_map > 0]
            self._value_map[new_map > 0] = new_value_map[new_map > 0]
            return
        elif self._fusion_type == "equal_weighting":
            # 等权重模式：更新的值总是当前值和新值的平均值
            print("VALUE MAP ABLATION:", self._fusion_type)
            self._map[self._map > 0] = 1
            new_map[new_map > 0] = 1
        else:
            assert self._fusion_type == "default", f"Unknown fusion type {self._fusion_type}"

        # 对于置信度低于阈值且低于现有地图置信度的值，将其置为0
        new_map_mask = np.logical_and(new_map < self._decision_threshold, new_map < self._map)
        new_map[new_map_mask] = 0

        if self._use_max_confidence:
            # 使用最大置信度：对于新地图中置信度更高的每个像素，
            # 用新值替换现有地图中的值
            higher_new_map_mask = new_map > self._map
            self._value_map[higher_new_map_mask] = values
            # 更新置信度地图
            self._map[higher_new_map_mask] = new_map[higher_new_map_mask]
        else:
            # 使用加权平均：根据当前和新置信度值的权重，
            # 更新每个像素的值和置信度
            confidence_denominator = self._map + new_map
            # 忽略除以零的警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weight_1 = self._map / confidence_denominator
                weight_2 = new_map / confidence_denominator

            # 将权重扩展为与价值图通道数相同的维度
            weight_1_channeled = np.repeat(np.expand_dims(weight_1, axis=2), self._value_channels, axis=2)
            weight_2_channeled = np.repeat(np.expand_dims(weight_2, axis=2), self._value_channels, axis=2)

            # 使用加权平均更新价值图和置信度地图
            self._value_map = self._value_map * weight_1_channeled + values * weight_2_channeled
            self._map = self._map * weight_1 + new_map * weight_2

            # 由于分母可能包含0值，将价值图和置信度地图中的NaN替换为0
            self._value_map = np.nan_to_num(self._value_map)
            self._map = np.nan_to_num(self._map)


def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    """Maps a value from one range to another.
    
    将一个值从一个范围映射到另一个范围
    
    Args:
        value: 要映射的值
        from_low: 输入范围的下界
        from_high: 输入范围的上界
        to_low: 输出范围的下界
        to_high: 输出范围的上界
        
    Returns:
        映射后的值
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def replay_from_dir() -> None:
    """
    从记录目录回放价值地图数据
    读取保存的参数和数据，逐帧重建价值地图
    """
    # 读取初始化参数
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    # 读取记录的数据
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # 创建价值地图对象
    v = ValueMap(**kwargs)

    # 获取并排序所有记录的图像路径
    sorted_keys = sorted(list(data.keys()))

    # 遍历每个记录的图像
    for img_path in sorted_keys:
        # 加载数据
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        values = np.array(data[img_path]["values"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # 更新地图
        v.update_map(
            values,
            depth,
            tf_camera_to_episodic,
            float(data[img_path]["min_depth"]),
            float(data[img_path]["max_depth"]),
            float(data[img_path]["fov"]),
        )

        # 可视化地图
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):  # 按q键退出
            break


if __name__ == "__main__":
    # 在回放模式下，执行回放函数
    if PLAYING:
        replay_from_dir()
        quit()

    # 创建一个单通道价值地图
    v = ValueMap(value_channels=1)
    # 读取示例深度图
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # 处理局部数据
    img = v._process_local_data(
        depth=depth,
        fov=np.deg2rad(79),
        min_depth=0.5,
        max_depth=5.0,
    )
    # 显示处理后的图像
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)

    # 测试数据
    num_points = 20

    # 定义一个方形路径的坐标和角度
    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    # 遍历路径上的点，更新地图
    for pt, angle in zip(points, angles):
        # 创建变换矩阵
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        # 更新地图
        v.update_map(
            np.array([1]),
            depth,
            tf,
            min_depth=0.5,
            max_depth=5.0,
            fov=np.deg2rad(79),
        )
        # 可视化地图
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):  # 按q键退出
            break
