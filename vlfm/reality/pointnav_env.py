# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from vlfm.reality.robots.bdsw_robot import BDSWRobot
from vlfm.reality.robots.camera_ids import SpotCamIds
from vlfm.utils.geometry_utils import (
    convert_to_global_frame,
    pt_from_rho_theta,
    rho_theta,
)


class PointNavEnv:
    """
    Gym environment for doing the PointNav task on the Spot robot in the real world.
    
    点导航(PointNav)任务环境类，用于在现实世界中使用Spot机器人执行导航任务
    提供类似于Gym的接口，包含reset、step、观察获取等功能
    """

    goal: Any = (None,)  # 目标位置，初始为空值，必须在reset()中设置
    info: Dict = {}  # 信息字典，可用于存储额外状态信息

    def __init__(
        self,
        robot: BDSWRobot,  # Boston Dynamics Spot包装机器人实例
        max_body_cam_depth: float = 3.5,  # 机身相机的最大深度值（米）
        max_lin_dist: float = 0.25,  # 每步最大线性移动距离（米）
        max_ang_dist: float = np.deg2rad(30),  # 每步最大角度变化（弧度，默认30度）
        time_step: float = 0.5,  # 每步时间（秒）
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化点导航环境
        
        Args:
            robot: Boston Dynamics Spot包装机器人实例
            max_body_cam_depth: 机身相机的最大深度值（米）
            max_lin_dist: 每步最大线性移动距离（米）
            max_ang_dist: 每步最大角度变化（弧度）
            time_step: 控制步长的时间间隔（秒）
        """
        self.robot = robot  # 存储机器人实例

        self._max_body_cam_depth = max_body_cam_depth  # 机身相机最大深度
        self._max_lin_dist = max_lin_dist  # 最大线性距离
        self._max_ang_dist = max_ang_dist  # 最大角度距离
        self._time_step = time_step  # 时间步长
        self._cmd_id: Union[None, Any] = None  # 当前命令ID，用于跟踪命令执行状态
        self._num_steps = 0  # 当前回合的步数计数器

    def reset(self, goal: Any, relative: bool = True, *args: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
        """
        重置环境并设置新的导航目标
        
        Args:
            goal: 目标位置坐标
            relative: 是否为相对于机器人当前位置的坐标（True）或全局坐标（False）
            
        Returns:
            初始观察数据字典
        """
        assert isinstance(goal, np.ndarray)  # 确保目标是numpy数组
        if relative:
            # 将目标从机器人坐标系转换到全局坐标系
            pos, yaw = self.robot.xy_yaw  # 获取机器人位置和朝向
            pos_w_z = np.array([pos[0], pos[1], 0.0])  # 添加Z坐标（设为0）
            goal_w_z = np.array([goal[0], goal[1], 0.0])  # 给目标也添加Z坐标（设为0）
            goal = convert_to_global_frame(pos_w_z, yaw, goal_w_z)[:2]  # 转换到全局坐标系，并去除Z坐标
        self.goal = goal  # 存储目标位置
        return self._get_obs()  # 返回初始观察数据

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作，移动机器人并返回新的观察结果
        
        Args:
            action: 动作字典，包含"linear"(线速度)和"angular"(角速度)或"rho_theta"(目标相对位置)
            
        Returns:
            观察数据、奖励、完成标志和信息字典的元组
        """
        # 如果有未完成的命令，等待其完成
        if self._cmd_id is not None:
            cmd_status = 0
            while cmd_status != 1:
                # 获取命令执行反馈
                feedback_resp = self.robot.spot.get_cmd_feedback(self._cmd_id)
                cmd_status = (
                    feedback_resp.feedback.synchronized_feedback
                ).mobility_command_feedback.se2_trajectory_feedback.status
                if cmd_status != 1:
                    time.sleep(0.1)  # 等待命令完成

        # 计算角度和线性位移
        ang_dist, lin_dist = self._compute_displacements(action)
        # 如果线速度和角速度都为0，认为任务完成
        done = action["linear"] == 0.0 and action["angular"] == 0.0
        print("ang/lin:", ang_dist, lin_dist)

        # 根据动作类型处理位置命令
        if "rho_theta" in action:
            # 如果提供了rho_theta（距离和角度），计算目标点坐标
            rho, theta = action["rho_theta"]
            x_pos, y_pos = pt_from_rho_theta(rho, theta)  # 从极坐标转换为笛卡尔坐标
            yaw = theta  # 目标朝向角
            print("RHO", rho)
        else:
            # 否则使用线性和角度位移
            x_pos = lin_dist
            y_pos = 0
            yaw = ang_dist

        # 如果完成，停止机器人
        if done:
            self.robot.command_base_velocity(0.0, 0.0)

        # 设置机器人基础位置
        self._cmd_id = self.robot.spot.set_base_position(
            x_pos=x_pos,  # X位置（前进/后退）
            y_pos=y_pos,  # Y位置（左/右）
            yaw=yaw,  # 旋转角度
            end_time=100,  # 命令结束时间（秒）
            relative=True,  # 相对于当前位置
            max_fwd_vel=0.3,  # 最大前进速度（米/秒）
            max_hor_vel=0.2,  # 最大水平速度（米/秒）
            max_ang_vel=np.deg2rad(60),  # 最大角速度（弧度/秒）
            disable_obstacle_avoidance=False,  # 启用障碍物避免
            blocking=False,  # 非阻塞调用
        )
        # 如果使用rho_theta控制，清空命令ID
        if "rho_theta" in action:
            self._cmd_id = None

        self._num_steps += 1  # 步数加1
        return self._get_obs(), 0.0, done, {}  # 返回观察、奖励、结束标志和空信息字典

    def _compute_velocities(self, action: Dict[str, Any]) -> Tuple[float, float]:
        """
        根据动作计算角速度和线速度
        
        Args:
            action: 动作字典
            
        Returns:
            角速度和线速度的元组
        """
        ang_dist, lin_dist = self._compute_displacements(action)  # 首先计算位移
        ang_vel = ang_dist / self._time_step  # 计算角速度
        lin_vel = lin_dist / self._time_step  # 计算线速度
        return ang_vel, lin_vel

    def _compute_displacements(self, action: Dict[str, Any]) -> Tuple[float, float]:
        """
        根据动作计算角度和线性位移
        
        Args:
            action: 动作字典，包含"angular"和"linear"键
            
        Returns:
            角度位移和线性位移的元组
        """
        displacements = []
        # 依次处理角度和线性动作
        for action_key, max_dist in (
            ("angular", self._max_ang_dist),
            ("linear", self._max_lin_dist),
        ):
            if action_key not in action:
                displacements.append(0.0)  # 如果没有提供该动作，使用0
                continue
            act_val = action[action_key]
            dist = np.clip(act_val, -1.0, 1.0)  # 将动作值限制在[-1, 1]范围内
            dist *= max_dist  # 缩放到最大位移
            displacements.append(dist)  # 添加到位移列表
        ang_dist, lin_dist = displacements  # 解包为角度和线性位移
        return ang_dist, lin_dist

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前环境的观察数据
        
        Returns:
            包含深度图像和目标位置信息的观察字典
        """
        return {
            "depth": self._get_nav_depth(),  # 获取导航用深度图
            "pointgoal_with_gps_compass": self._get_rho_theta(),  # 获取目标的距离和角度
        }

    def _get_nav_depth(self) -> np.ndarray:
        """
        获取用于导航的深度图像
        合并前方左右两个深度相机的图像
        
        Returns:
            处理后的深度图像
        """
        # 获取前方左右两个深度相机的图像
        images = self.robot.get_camera_images([SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH])
        # 水平拼接两个图像
        img = np.hstack([images[SpotCamIds.FRONTRIGHT_DEPTH], images[SpotCamIds.FRONTLEFT_DEPTH]])
        # 归一化深度图像
        img = self._norm_depth(img)
        return img

    def _norm_depth(self, depth: np.ndarray, max_depth: Optional[float] = None, scale: bool = True) -> np.ndarray:
        """
        归一化深度图像
        将原始深度值转换为[0,1]范围的归一化值
        
        Args:
            depth: 原始深度图像
            max_depth: 最大深度值，如果未提供则使用默认值
            scale: 是否需要将原始值从毫米转换为米
            
        Returns:
            归一化后的深度图像
        """
        if max_depth is None:
            max_depth = self._max_body_cam_depth  # 使用默认最大深度

        norm_depth = depth.astype(np.float32)  # 转换为浮点型

        if scale:
            # 从毫米转换为米（对于uint16类型的原始深度图）
            norm_depth = norm_depth / 1000.0

        # 归一化到[0,1]范围
        norm_depth = np.clip(norm_depth, 0.0, max_depth) / max_depth

        return norm_depth

    def _get_rho_theta(self) -> np.ndarray:
        """
        计算当前位置到目标的距离和角度
        
        Returns:
            包含距离(rho)和角度(theta)的numpy数组
        """
        curr_pos, yaw = self.robot.xy_yaw  # 获取当前位置和朝向
        r_t = rho_theta(curr_pos, yaw, self.goal)  # 计算到目标的距离和角度
        return np.array(r_t)
