# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union

import cv2
import numpy as np
import open3d as o3d

from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
)


class ObjectPointCloudMap:
    """
    对象点云地图类
    用于存储和维护场景中检测到的各类对象的3D点云
    """
    clouds: Dict[str, np.ndarray] = {}  # 存储不同类别对象的点云，键为对象类名，值为点云数据
    use_dbscan: bool = True  # 是否使用DBSCAN聚类算法过滤点云

    def __init__(self, erosion_size: float) -> None:
        """
        初始化对象点云地图
        
        Args:
            erosion_size: 掩码腐蚀的大小，用于减少边缘噪声
        """
        self._erosion_size = erosion_size
        self.last_target_coord: Union[np.ndarray, None] = None  # 最后一次目标坐标，用于稳定性跟踪

    def reset(self) -> None:
        """
        重置地图，清空所有存储的点云数据
        """
        self.clouds = {}
        self.last_target_coord = None

    def has_object(self, target_class: str) -> bool:
        """
        检查指定类别的对象是否存在于地图中
        
        Args:
            target_class: 目标对象类别名
            
        Returns:
            是否存在该类别的点云数据
        """
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Updates the object map with the latest information from the agent.
        
        使用代理提供的最新信息更新对象地图。
        
        Args:
            object_name: 对象类别名称
            depth_img: 深度图像
            object_mask: 对象分割掩码
            tf_camera_to_episodic: 从相机坐标系到全局坐标系的变换矩阵
            min_depth: 最小有效深度值
            max_depth: 最大有效深度值
            fx: 相机x方向焦距
            fy: 相机y方向焦距
        """
        # 从深度图和掩码中提取对象的点云
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return  # 如果未能提取到点云，直接返回

        # 对于质量不佳或位于边缘的检测结果，我们为其点云最后一列分配随机值，
        # 以便之后能识别来自同一检测的点
        if too_offset(object_mask):  # 检查对象是否位于图像边缘
            # 边缘检测使用随机值标记所有点
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            # 标记距离相机太远的点为超出范围
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 预留5%余量
            # 将within_range中值为1的视为在范围内，值为0的视为超出范围
            # 超出范围的点使用随机值标记，以便后续识别
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
            
        # 将点云从相机坐标系转换到全局坐标系
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        # 将范围标记追加到点云最后一列
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        # 获取相机当前位置
        curr_position = tf_camera_to_episodic[:3, 3]
        # 找到点云中离相机最近的点
        closest_point = self._get_closest_point(global_cloud, curr_position)
        # 计算最近点到相机的距离
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0:
            # 对象太近，可能是噪声，不可信
            return

        # 将新的点云添加到现有点云中
        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        """
        获取指定类别中最适合作为导航目标的对象位置坐标
        通过计算最近点并应用平滑策略避免目标位置抖动
        
        Args:
            target_class: 目标对象的类别
            curr_position: 当前机器人位置
            
        Returns:
            最佳目标对象的2D坐标
        """
        # 获取目标类别的点云
        target_cloud = self.get_target_cloud(target_class)

        # 获取离当前位置最近的点（仅取XY坐标）
        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]

        if self.last_target_coord is None:
            # 首次检测，直接使用最近点
            self.last_target_coord = closest_point_2d
        else:
            # 在以下情况下不更新last_target_coord：
            # 1. 最近的点仅有微小变化
            # 2. 最近的点有轻微变化，但机器人距离太远，变化不太重要
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                # 最近点只有微小变化，保持稳定性
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0:
                # 最近点有轻微变化，但机器人距离太远，变化不太重要
                return self.last_target_coord
            else:
                # 变化显著或距离较近，更新目标坐标
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.
        
        该方法移除所有原本被检测为超出范围但现在在视野范围内的点云
        这是一种启发式方法，用于抑制我们现在确认不是实际目标对象的短暂误检
        
        Args:
            tf_camera_to_episodic: 从相机坐标系到全局坐标系的变换矩阵
            max_depth: 相机的最大深度范围
            cone_fov: 相机的视野角度
        """
        # 获取相机位置和朝向
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        # 对每个存储的对象类别执行操作
        for obj in self.clouds:
            # 检查哪些点在当前视野锥体内
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,  # 使用一半的最大深度作为范围
                self.clouds[obj],
            )
            # 获取所有唯一的范围ID
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # 检测原本就在范围内，保留
                    continue
                # 移除所有具有相同range_id的点
                # 这些点原本被标记为超出范围，但现在在视野内没有观察到，可能是误检
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        """
        获取目标类别的点云，并筛选出在范围内的点
        
        Args:
            target_class: 目标对象类别
            
        Returns:
            目标对象的点云数据
        """
        # 复制点云数据以避免修改原始数据
        target_cloud = self.clouds[target_class].copy()
        # 确定是否存在范围内的点
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # 如果存在范围内的点，则过滤掉范围外的点
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        """
        从深度图和对象掩码中提取对象的点云
        
        Args:
            depth: 深度图像
            object_mask: 对象分割掩码
            min_depth: 最小有效深度
            max_depth: 最大有效深度
            fx: 相机X方向焦距
            fy: 相机Y方向焦距
            
        Returns:
            提取的对象点云数据
        """
        # 将掩码值缩放到0-255
        final_mask = object_mask * 255
        # 对掩码进行腐蚀操作，减少边缘噪声
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        # 处理深度图，将无效值(0)设为远处(1)
        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # 将所有空洞(0)设置为远处(1)
        # 将归一化深度值映射回实际范围
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        # 根据深度图和掩码生成点云
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        # 随机减采样点云，控制点数量
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            # 使用DBSCAN聚类算法过滤点云，去除噪点
            cloud = open3d_dbscan_filtering(cloud)

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        """
        获取点云中离当前位置最近的点
        根据use_dbscan设置使用不同的查找策略
        
        Args:
            cloud: 点云数据
            curr_position: 当前位置坐标
            
        Returns:
            离当前位置最近的点坐标
        """
        ndim = curr_position.shape[0]  # 当前位置的维度
        if self.use_dbscan:
            # 使用简单欧氏距离找最近点
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            # 使用更复杂的策略，考虑一部分靠近的点
            # 计算每个点到参考点的欧氏距离
            if ndim == 2:
                # 如果是2D位置，添加默认高度值
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # 使用argsort获取距离排序的索引
            sorted_indices = np.argsort(distances)

            # 获取距离最近的前25%的点
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                # 返回中间点，增加稳定性
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point


def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    """
    使用DBSCAN聚类算法过滤点云数据
    保留最大的点云簇，去除噪声和小簇
    
    Args:
        points: 输入点云数据
        eps: DBSCAN的邻域半径参数
        min_points: 一个核心点邻域内最少点数
        
    Returns:
        过滤后的点云数据，如果未检测到有效簇，则返回空数组
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 执行DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # 统计每个簇中的点数量
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # 排除噪声点（标签为-1）
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # 只检测到噪声
        return np.array([])

    # 找出最大非噪声簇的标签
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # 获取最大簇中所有点的索引
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # 获取最大簇中的点
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    """Visualizes an array of 3D points and saves the visualization as a PNG image.
    
    可视化3D点云并将结果保存为PNG图像
    
    Args:
        point_cloud (np.ndarray): 形状为(N, 3)的3D点数组
        save_path (str): 保存PNG图像的路径
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(save_path)
    plt.close()


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.
    
    从给定的3D点数组中随机选择一个子集
    如果原始数组大小小于指定大小，则直接返回原始数组
    这个函数用于降低点云密度，提高处理效率
    
    Args:
        points (numpy array): 3D点的numpy数组
        size (int): 期望的子数组大小
        
    Returns:
        numpy array: 原始点数组的子集
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def too_offset(mask: np.ndarray) -> bool:
    """
    This will return true if the entire bounding rectangle of the mask is either on the
    left or right third of the mask. This is used to determine if the object is too far
    to the side of the image to be a reliable detection.
    
    检查掩码是否位于图像的左侧或右侧边缘
    用于判断对象是否太靠近图像边缘而导致检测不可靠
    
    Args:
        mask (numpy array): 表示对象掩码的2D numpy数组
        
    Returns:
        bool: 如果对象太靠近边缘则返回True，否则返回False
    """
    # Find the bounding rectangle of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Calculate the thirds of the mask
    third = mask.shape[1] // 3

    # Check if the entire bounding rectangle is in the left or right third of the mask
    if x + w <= third:
        # Check if the leftmost point is at the edge of the image
        # return x == 0
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        # Check if the rightmost point is at the edge of the image
        # return x + w == mask.shape[1]
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
