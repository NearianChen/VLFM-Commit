# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, List

import cv2
import numpy as np
import torch
from depth_camera_filtering import filter_depth
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_policy import BasePolicy
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix


@baseline_registry.register_policy  # 这个装饰器用于将自定义策略注册到Habitat的策略注册表中
class ActionReplayPolicy(BasePolicy):
    """
    动作回放策略类
    用于从记录的动作序列中回放动作，同时保存观察结果和地图信息
    """
    def __init__(
        self,
        forward_step_size: float,    # 前进步长
        turn_angle: float,           # 转弯角度
        min_obstacle_height: float,  # 障碍物最小高度
        max_obstacle_height: float,  # 障碍物最大高度
        obstacle_map_area_threshold: float,  # 障碍地图区域阈值
        agent_radius: float,         # 代理半径
        hole_area_thresh: int,       # 洞区域阈值
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化动作回放策略
        
        Args:
            forward_step_size: 代理前进的步长（米）
            turn_angle: 代理转弯的角度（度）
            min_obstacle_height: 视为障碍物的最小高度
            max_obstacle_height: 视为障碍物的最大高度
            obstacle_map_area_threshold: 障碍地图区域阈值
            agent_radius: 代理半径，用于障碍物膨胀
            hole_area_thresh: 要填充的深度图中的洞的面积阈值
        """
        super().__init__()
        # 确保环境变量中设置了动作记录目录
        assert "VLFM_RECORD_ACTIONS_DIR" in os.environ, "Must set VLFM_RECORD_ACTIONS_DIR"
        # 获取记录目录路径
        self._dir = os.environ["VLFM_RECORD_ACTIONS_DIR"]
        # 动作文件路径
        filepath = os.path.join(self._dir, "actions.txt")
        # 读取记录的动作序列
        with open(filepath, "r") as f:
            self._actions = [int(i) for i in f.readlines()]
        
        # 计算转弯和前进动作的重复次数，用于调整动作粒度
        turn_repeat = int(30 / turn_angle)  # 将标准30度转弯调整为配置中的角度
        step_repeat = int(0.25 / forward_step_size)  # 将标准0.25米步长调整为配置中的步长
        
        # 对转弯动作(2和3)进行重复，以匹配期望的转弯角度
        for turn_type in [2, 3]:  # 2表示左转，3表示右转
            self._actions = repeat_elements(self._actions, turn_type, turn_repeat)
        # 对前进动作(1)进行重复，以匹配期望的步长
        self._actions = repeat_elements(self._actions, 1, step_repeat)
        
        # 创建图像存储目录
        img_dir = os.path.join(self._dir, "imgs")
        # 如果目录不存在，则创建
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # 初始化当前动作索引
        self.curr_idx = 0

        # 设置相机高度
        self._camera_height = 0.88
        # 初始化障碍物地图
        self._obstacle_map = ObstacleMap(
            min_height=min_obstacle_height,         # 障碍物最小高度
            max_height=max_obstacle_height,         # 障碍物最大高度
            area_thresh=obstacle_map_area_threshold,  # 区域阈值
            agent_radius=agent_radius,              # 代理半径
            hole_area_thresh=hole_area_thresh,      # 洞区域阈值
            pixels_per_meter=50,                    # 每米像素数
            size=2500,                              # 地图尺寸
        )
        # 设置障碍物地图中代理半径填充的颜色为黑色
        self._obstacle_map.radius_padding_color = (0, 0, 0)
        # 设置相机视场角（弧度制，对应79度）
        self._camera_fov_rad = np.deg2rad(79)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,         # 配置对象
        observation_space: spaces.Dict,  # 观察空间
        action_space: spaces.Space,      # 动作空间
        **kwargs: Any,
    ) -> "ActionReplayPolicy":
        """
        从配置创建策略实例的类方法
        
        Args:
            config: 包含所需参数的配置对象
            observation_space: 观察空间定义
            action_space: 动作空间定义
            
        Returns:
            配置好的ActionReplayPolicy实例
        """
        # 获取策略配置部分
        policy_cfg = config.habitat_baselines.rl.policy

        # 使用配置参数创建实例
        return cls(
            forward_step_size=config.habitat.simulator.forward_step_size,  # 前进步长
            turn_angle=config.habitat.simulator.turn_angle,                # 转弯角度
            min_obstacle_height=policy_cfg.min_obstacle_height,            # 障碍物最小高度
            max_obstacle_height=policy_cfg.max_obstacle_height,            # 障碍物最大高度
            obstacle_map_area_threshold=policy_cfg.obstacle_map_area_threshold,  # 区域阈值
            agent_radius=policy_cfg.agent_radius,                          # 代理半径
            hole_area_thresh=policy_cfg.hole_area_thresh,                  # 洞区域阈值
        )

    def act(
        self,
        observations: TensorDict,     # 观察数据
        rnn_hidden_states: Any,       # RNN隐藏状态
        prev_actions: torch.Tensor,   # 前一个动作
        masks: torch.Tensor,          # 掩码
        deterministic: bool = False,  # 是否确定性行为
    ) -> PolicyActionData:
        """
        执行策略动作方法，从记录的动作序列中选择动作
        同时保存当前观察的图像、位置和地图信息
        
        Args:
            observations: 当前观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一个执行的动作
            masks: 用于RNN的掩码
            deterministic: 是否使用确定性行为
            
        Returns:
            包含选择的动作和相关信息的PolicyActionData对象
        """
        # 保存RGB和深度图像
        rgb = observations["rgb"][0].cpu().numpy()  # 获取RGB图像数据
        depth = observations["depth"][0].cpu().numpy()  # 获取深度图像数据
        
        # 构建保存路径
        rgb_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_rgb.png",  # 使用索引命名文件
        )
        depth_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_depth.png",
        )

        # 保存图像
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 转换为BGR格式（OpenCV格式）
        cv2.imwrite(rgb_path, bgr)  # 保存RGB图像
        depth_int = (depth * 255).astype("uint8")  # 将深度图归一化为8位
        cv2.imwrite(depth_path, depth_int)  # 保存深度图像

        # 记录位置和朝向
        x, y = observations["gps"][0].cpu().numpy()  # 获取GPS位置
        csv_data = [
            str(x),  # X坐标
            str(-y),  # Y坐标（翻转Y轴）
            str(observations["compass"][0].cpu().item()),  # 罗盘朝向
            str(observations["heading"][0].item()),  # 航向角
        ]
        csv_line = ",".join(csv_data)  # 组合为CSV行
        filepath = os.path.join(
            self._dir,
            "position_yaws.csv",  # 位置和朝向记录文件
        )

        # 如果文件不存在，创建文件并写入表头
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write("x,y,compass,heading\n")
        # 追加当前位置和朝向数据
        with open(filepath, "a") as f:
            f.write(f"{csv_line}\n")

        # 更新障碍物地图
        # 计算相机焦距，基于图像宽度和视场角
        image_width = depth.shape[1]
        fx = fy = image_width / (2 * np.tan(self._camera_fov_rad / 2))

        # 获取相机位置和朝向
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        # 过滤深度图像，去噪
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS使西方向为负，所以翻转Y轴
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]  # 提取XY坐标
        # 创建从相机到全局坐标系的变换矩阵
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # 使用当前观察更新障碍物地图
        self._obstacle_map.update_map(
            depth,
            tf_camera_to_episodic,
            0.5,  # 最小深度
            5.0,  # 最大深度
            fx,   # X方向焦距
            fy,   # Y方向焦距
            self._camera_fov_rad,  # 相机视场角
        )
        # 更新代理轨迹
        self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        # 构建边界地图保存路径
        frontier_map_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_frontier_map.png",
        )
        # 保存可视化的障碍物地图
        cv2.imwrite(frontier_map_path, self._obstacle_map.visualize())

        # 获取当前索引的动作并转换为张量
        action = torch.tensor([self._actions[self.curr_idx]], dtype=torch.long)
        # 构建动作数据对象
        action_data = PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[{}],  # 策略信息，这里为空
        )
        # 更新索引，准备下一个动作
        self.curr_idx += 1

        return action_data


def repeat_elements(lst: List[int], element: int, repeat_count: int) -> List[int]:
    """
    重复列表中指定元素的辅助函数
    
    Args:
        lst: 输入的整数列表
        element: 要重复的元素值
        repeat_count: 重复次数
        
    Returns:
        处理后的列表，其中指定元素被重复指定次数
    """
    new_list = []
    for i in lst:
        if i == element:  # 如果是指定元素
            new_list.extend([i] * repeat_count)  # 重复添加
        else:
            new_list.append(i)  # 其他元素直接添加
    return new_list
