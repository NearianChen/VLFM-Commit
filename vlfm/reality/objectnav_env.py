# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from depth_camera_filtering import filter_depth

from vlfm.reality.pointnav_env import PointNavEnv
from vlfm.reality.robots.camera_ids import SpotCamIds
from vlfm.utils.geometry_utils import get_fov, wrap_heading
from vlfm.utils.img_utils import reorient_rescale_map, resize_images

LEFT_CROP = 124  # 左侧裁剪像素值
RIGHT_CROP = 60  # 右侧裁剪像素值
NOMINAL_ARM_POSE = np.deg2rad([0, -170, 120, 0, 55, 0])  # 机械臂的标准姿态（弧度制）

VALUE_MAP_CAMS = [
    # 用于构建价值地图的相机列表
    # 目前仅使用手部彩色相机，其他相机被注释掉
    # SpotCamIds.BACK_FISHEYE,
    # SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
    # SpotCamIds.LEFT_FISHEYE,
    # SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    # SpotCamIds.RIGHT_FISHEYE,
    # SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.HAND_COLOR,
]

POINT_CLOUD_CAMS = [
    # 用于构建点云和障碍物地图的相机列表
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
]

ALL_CAMS = list(set(VALUE_MAP_CAMS + POINT_CLOUD_CAMS))  # 所有需要使用的相机的合集


class ObjectNavEnv(PointNavEnv):
    """
    Gym environment for doing the ObjectNav task on the Spot robot in the real world.
    
    在真实世界中使用Spot机器人执行目标对象导航任务的Gym环境
    """

    tf_episodic_to_global: np.ndarray = np.eye(4)  # 从回合坐标系到全局坐标系的变换矩阵，需要在reset()中设置
    tf_global_to_episodic: np.ndarray = np.eye(4)  # 从全局坐标系到回合坐标系的变换矩阵，需要在reset()中设置
    episodic_start_yaw: float = float("inf")  # 回合开始时的机器人朝向，需要在reset()中设置
    target_object: str = ""  # 目标对象名称，需要在reset()中设置

    def __init__(self, max_gripper_cam_depth: float, *args: Any, **kwargs: Any) -> None:
        """
        初始化目标对象导航环境
        
        Args:
            max_gripper_cam_depth: 机器人夹持器相机的最大深度值（米）
            *args, **kwargs: 传递给父类的额外参数
        """
        super().__init__(*args, **kwargs)
        self._max_gripper_cam_depth = max_gripper_cam_depth
        # 获取当前日期和时间，用于创建唯一的可视化目录
        now = datetime.now()
        # 格式化为MM-DD-HH-MM-SS格式的字符串
        date_string = now.strftime("%m-%d-%H-%M-%S")
        self._vis_dir = f"{date_string}"
        # 创建可视化目录用于存储导航过程中的图像
        os.makedirs(f"vis/{self._vis_dir}", exist_ok=True)

    def reset(self, goal: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        重置环境并设置新的目标对象
        
        Args:
            goal: 目标对象名称（字符串）
            *args, **kwargs: 额外参数
            
        Returns:
            初始观察数据字典
        """
        assert isinstance(goal, str)  # 确保目标是字符串类型
        self.target_object = goal  # 设置目标对象
        # 获取从机器人起始位置到全局坐标系的变换矩阵
        self.tf_episodic_to_global: np.ndarray = self.robot.get_transform()
        self.tf_episodic_to_global[2, 3] = 0.0  # 将z坐标置为0，使变换矩阵只在水平面内
        # 计算从全局到回合坐标系的逆变换
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        # 记录回合开始时的机器人朝向
        self.episodic_start_yaw = self.robot.xy_yaw[1]
        return self._get_obs()  # 返回初始观察数据

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作，更新环境状态
        包括基础移动、机械臂控制和可视化处理
        
        Args:
            action: 包含动作信息的字典，包含机器人移动和臂部控制信息
            
        Returns:
            观察数据、奖励、完成标志和信息字典的元组
        """
        # 处理可视化信息
        vis_imgs = []
        time_id = time.time()  # 使用当前时间作为图像文件名
        for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
            img = cv2.cvtColor(action["info"][k], cv2.COLOR_RGB2BGR)  # 转换颜色空间
            cv2.imwrite(f"vis/{self._vis_dir}/{time_id}_{k}.png", img)  # 保存单独的图像
            if "map" in k:
                img = reorient_rescale_map(img)  # 重新调整地图方向和大小
            if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
                # 在深度图上添加"目标未检测到"的文本
                text = "Target not currently detected"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                cv2.putText(
                    img,
                    text,
                    (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )
            vis_imgs.append(img)
        # 将所有图像水平拼接并保存
        vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
        cv2.imwrite(f"vis/{self._vis_dir}/{time_id}.jpg", vis_img)
        # 根据环境变量决定是否显示可视化窗口
        if os.environ.get("ZSOS_DISPLAY", "0") == "1":
            cv2.imshow("Visualization", cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(1)

        # 处理不同的动作类型
        if action["arm_yaw"] == -1:
            # -1表示使用基类（PointNavEnv）的step方法，只移动机器人基座
            return super().step(action)

        if action["arm_yaw"] == 0:
            # 0表示将夹持器移动到前方指定位置
            cmd_id = self.robot.spot.move_gripper_to_point(np.array([0.35, 0.0, 0.3]), np.deg2rad([0.0, 0.0, 0.0]))
            self.robot.spot.block_until_arm_arrives(cmd_id, timeout_sec=1.5)  # 等待机械臂到达位置
        else:
            # 其他值表示旋转机械臂到指定角度
            new_pose = np.array(NOMINAL_ARM_POSE)  # 复制标准姿态
            new_pose[0] = action["arm_yaw"]  # 替换第一个关节角度
            self.robot.set_arm_joints(new_pose, travel_time=0.5)  # 设置机械臂关节角度
            time.sleep(0.75)  # 等待机械臂到达位置
        done = False  # 目标未完成
        self._num_steps += 1  # 增加步数计数

        return self._get_obs(), 0.0, done, {}  # 返回观察、奖励、结束标志和空信息字典

    def _get_obs(self) -> Dict[str, Any]:
        """
        获取当前环境的完整观察数据
        
        Returns:
            包含机器人位置、朝向、相机图像和目标对象信息的观察字典
        """
        robot_xy, robot_heading = self._get_gps(), self._get_compass()  # 获取位置和朝向
        nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = self._get_camera_obs()  # 获取相机观察数据
        return {
            "nav_depth": nav_depth,  # 导航用深度图
            "robot_xy": robot_xy,  # 机器人XY坐标
            "robot_heading": robot_heading,  # 机器人朝向角
            "objectgoal": self.target_object,  # 目标对象名称
            "obstacle_map_depths": obstacle_map_depths,  # 障碍物地图深度数据
            "value_map_rgbd": value_map_rgbd,  # 价值地图RGB-D数据
            "object_map_rgbd": object_map_rgbd,  # 对象地图RGB-D数据
        }

    def _get_camera_obs(self) -> Tuple[np.ndarray, List, List, List]:
        """
        Poll all necessary cameras on the robot and return their images, focal lengths,
        and transforms to the global frame.
        
        获取机器人所有必要相机的数据，包括图像、焦距和坐标变换
        处理并整合多个相机的观察结果，用于导航、障碍物检测和目标识别
        
        Returns:
            导航深度图、障碍物地图深度数据、价值地图RGB-D数据和对象地图RGB-D数据的元组
        """
        srcs: List[str] = ALL_CAMS  # 所有需要采集数据的相机源
        cam_data = self.robot.get_camera_data(srcs)  # 获取所有相机数据
        
        # 处理每个相机的数据
        for src in ALL_CAMS:
            # 将相机坐标系从全局变换到回合坐标系
            tf = self.tf_global_to_episodic @ cam_data[src]["tf_camera_to_global"]
            # 应用旋转矩阵，将相机约定映射到XYZ约定
            rotation_matrix = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
            cam_data[src]["tf_camera_to_global"] = np.dot(tf, rotation_matrix)

            img = cam_data[src]["image"]

            # 归一化和过滤深度图像；但暂不过滤导航用深度图
            if img.dtype == np.uint16:  # 深度图通常为16位整数
                if "hand" in src:
                    max_depth = self._max_gripper_cam_depth  # 使用夹持器相机的最大深度
                else:
                    max_depth = self._max_body_cam_depth  # 使用机身相机的最大深度
                img = self._norm_depth(img, max_depth=max_depth)  # 深度归一化
                # 前方相机用于导航，暂不过滤
                if "front" not in src:
                    img = filter_depth(img, blur_type=None, recover_nonzero=False)  # 过滤其他深度图
                cam_data[src]["image"] = img

            # 处理彩色图像，确保统一格式
            if img.dtype == np.uint8:
                if img.ndim == 2 or img.shape[2] == 1:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 灰度转RGB
                else:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

        min_depth = 0  # 最小深度值

        # 准备对象地图输出，包含RGB、深度、变换矩阵和相机参数
        src = SpotCamIds.HAND_COLOR  # 使用手部彩色相机
        rgb = cam_data[src]["image"]
        hand_depth = np.ones(rgb.shape[:2], dtype=np.float32)  # 创建全1深度图（将由深度估计模型替换）
        tf = cam_data[src]["tf_camera_to_global"]
        max_depth = self._max_gripper_cam_depth
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]  # 相机焦距
        object_map_rgbd = [(rgb, hand_depth, tf, min_depth, max_depth, fx, fy)]

        # 处理导航用深度图，需要前方两个相机的图像并旋转至正立
        f_left = SpotCamIds.FRONTLEFT_DEPTH
        f_right = SpotCamIds.FRONTRIGHT_DEPTH
        nav_cam_data = self.robot.reorient_images({k: cam_data[k]["image"] for k in [f_left, f_right]})
        nav_depth = np.hstack([nav_cam_data[f_right], nav_cam_data[f_left]])  # 水平拼接两个图像
        nav_depth = filter_depth(nav_depth, blur_type=None, set_black_value=1.0)  # 过滤深度图

        # 准备障碍物地图输出，包含深度、变换矩阵和相机参数
        obstacle_map_depths = []
        # 在初始阶段使用所有相机，后期仅使用前方相机以提高效率
        if self._num_steps <= 10:
            srcs = POINT_CLOUD_CAMS.copy()  # 初始阶段使用所有点云相机
        else:
            srcs = POINT_CLOUD_CAMS[:2]  # 后期仅使用前方相机
        for src in srcs:
            depth = cam_data[src]["image"]
            fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
            tf = cam_data[src]["tf_camera_to_global"]
            # 根据相机类型选择适当的方向计算FOV
            if src in [SpotCamIds.FRONTLEFT_DEPTH, SpotCamIds.FRONTRIGHT_DEPTH]:
                fov = get_fov(fy, depth.shape[0])  # 前方相机使用垂直FOV
            else:
                fov = get_fov(fx, depth.shape[1])  # 其他相机使用水平FOV
            src_data = (depth, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
            obstacle_map_depths.append(src_data)

        # 添加手部相机的变换信息，用于探索地图更新
        tf = cam_data[SpotCamIds.HAND_COLOR]["tf_camera_to_global"]
        fx, fy = (
            cam_data[SpotCamIds.HAND_COLOR]["fx"],
            cam_data[SpotCamIds.HAND_COLOR]["fy"],
        )
        fov = get_fov(fx, cam_data[src]["image"].shape[1])
        src_data = (None, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
        obstacle_map_depths.append(src_data)

        # 准备价值地图输出，包含RGB、深度、变换矩阵和相机参数
        value_map_rgbd = []
        value_cam_srcs: List[str] = VALUE_MAP_CAMS + ["hand_depth_estimated"]
        # RGB相机在偶数索引，深度相机在奇数索引
        value_rgb_srcs = value_cam_srcs[::2]
        value_depth_srcs = value_cam_srcs[1::2]
        for rgb_src, depth_src in zip(value_rgb_srcs, value_depth_srcs):
            rgb = cam_data[rgb_src]["image"]
            # 如果是估计的手部深度，使用之前准备的全1深度图
            if depth_src == "hand_depth_estimated":
                depth = hand_depth
            else:
                depth = cam_data[src]["image"]
            fx = cam_data[rgb_src]["fx"]
            tf = cam_data[rgb_src]["tf_camera_to_global"]
            fov = get_fov(fx, rgb.shape[1])
            src_data = (rgb, depth, tf, min_depth, max_depth, fov)  # type: ignore
            value_map_rgbd.append(src_data)

        return nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd

    def _get_gps(self) -> np.ndarray:
        """
        Get the (x, y) position of the robot's base in the episode frame. x is forward,
        y is left.
        
        获取机器人基座在回合坐标系中的(x, y)位置
        其中x轴朝前，y轴朝左
        
        Returns:
            机器人在回合坐标系中的XY坐标
        """
        global_xy = self.robot.xy_yaw[0]  # 获取全局坐标系中的XY位置
        start_xy = self.tf_episodic_to_global[:2, 3]  # 获取回合起点在全局坐标系中的XY位置
        offset = global_xy - start_xy  # 计算相对于起点的偏移量
        # 构建旋转矩阵，将全局坐标系中的偏移旋转到回合坐标系中
        rotation_matrix = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        episodic_xy = rotation_matrix @ offset  # 应用旋转矩阵得到回合坐标系下的位置
        return episodic_xy

    def _get_compass(self) -> float:
        """
        Get the yaw of the robot's base in the episode frame. Yaw is measured in radians
        counterclockwise from the z-axis.
        
        获取机器人基座在回合坐标系中的朝向角
        朝向角以弧度计，从z轴逆时针测量
        
        Returns:
            机器人在回合坐标系中的朝向角（弧度）
        """
        global_yaw = self.robot.xy_yaw[1]  # 获取全局坐标系中的朝向角
        episodic_yaw = wrap_heading(global_yaw - self.episodic_start_yaw)  # 将全局朝向转换为回合坐标系中的朝向
        return episodic_yaw
