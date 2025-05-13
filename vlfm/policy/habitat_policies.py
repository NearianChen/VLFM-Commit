# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3

# Habitat场景数据集HM3D中对象ID到名称的映射列表
HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
# Habitat场景数据集MP3D中对象ID到名称的映射列表
# 使用|分隔的多个名称表示同一类对象的不同表述方式
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    """
    定义Habitat环境中的动作ID常量
    以PyTorch张量形式存储，方便直接作为策略输出使用
    """
    STOP = torch.tensor([[0]], dtype=torch.long)  # 停止动作
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)  # 前进动作
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)  # 左转动作
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)  # 右转动作


class HabitatMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    
    这个Python混入类只包含在Habitat环境中运行BaseObjectNavPolicy所需的代码
    （而不是在现实世界等环境中）。它将为任何父类（BaseObjectNavPolicy的子类）
    提供在Habitat中运行所需的方法。
    """

    _stop_action: Tensor = TorchActionIDs.STOP  # 停止动作张量
    _start_yaw: Union[float, None] = None  # 初始朝向角，必须由_reset()方法设置
    _observations_cache: Dict[str, Any] = {}  # 观察数据缓存
    _policy_info: Dict[str, Any] = {}  # 策略信息字典
    _compute_frontiers: bool = False  # 是否计算边界点

    def __init__(
        self,
        camera_height: float,  # 相机高度
        min_depth: float,      # 最小深度值
        max_depth: float,      # 最大深度值
        camera_fov: float,     # 相机视场角（度）
        image_width: int,      # 图像宽度
        dataset_type: str = "hm3d",  # 数据集类型
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化Habitat混入类
        
        Args:
            camera_height: 相机距地面高度（米）
            min_depth: 深度图最小值（米）
            max_depth: 深度图最大值（米）
            camera_fov: 相机水平视场角（度）
            image_width: 图像宽度（像素）
            dataset_type: 数据集类型，支持"hm3d"或"mp3d"
        """
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        # 将视场角从度转换为弧度
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        # 根据视场角和图像宽度计算相机焦距
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        """
        从配置创建HabitatMixin实例的类方法
        
        Args:
            config: Habitat配置对象
            
        Returns:
            配置好的HabitatMixin实例
        """
        # 获取策略配置
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy
        # 从配置中提取所有需要的参数
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  # type: ignore

        # 在Habitat中，我们需要相机高度来生成相机变换矩阵
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

        # 同步映射的最小/最大深度值与Habitat配置
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width

        # 只在需要保存视频时进行可视化
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        # 根据数据集路径推断数据集类型
        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData
        
        将对象ID转换为字符串名称，返回策略动作数据
        
        Args:
            observations: 观察数据字典
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步动作
            masks: 重置掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            包含动作和策略信息的PolicyActionData对象
        """
        # 获取目标对象ID
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        # 将TensorDict转换为普通字典
        obs_dict = observations.to_tree()
        # 根据数据集类型将对象ID转换为对象名称
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
            # 为非COCO对象设置描述提示，使用所有可能的对象名称
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
            
        # 调用父类的act方法计算动作
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            # 如果到达地图边缘，则停止
            action = self._stop_action
            
        # 返回策略动作数据
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],  # 包含策略信息，用于可视化和分析
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning
        
        通过左转12次，每次30度，获取环境的360度全景
        初始化阶段让机器人原地旋转以了解周围环境
        
        Returns:
            左转动作
        """
        # 在前11步执行初始化（旋转），完成后设置done_initializing为True
        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        """
        重置策略状态
        调用父类的重置方法并清空初始朝向记录
        """
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging
        
        获取用于日志记录的策略信息
        
        Args:
            detections: 对象检测结果
            
        Returns:
            包含策略信息的字典
        """
        # 获取父类的策略信息
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        # 如果不需要可视化，直接返回
        if not self._visualize:  # type: ignore
            return info

        # 记录初始朝向角
        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict) -> None:
        """Caches the rgb, depth, and camera transform from the observations.
        
        缓存RGB图像、深度图像和相机变换等观察数据
        
        Args:
           observations: 当前时间步的观察数据
        """
        # 如果缓存已有数据，则不重复处理
        if len(self._observations_cache) > 0:
            return
            
        # 提取RGB图像和深度图像
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        # 提取位置和朝向
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        # 过滤深度图像，去除噪声
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        
        # Habitat的GPS中西方为负数，因此翻转Y轴
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        # 创建从相机到全局坐标系的变换矩阵
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # 如果需要计算边界，则更新障碍物地图
        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            # 否则尝试直接从观察中获取边界信息
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        # 构建观察缓存字典
        self._observations_cache = {
            "frontier_sensor": frontiers,  # 边界点
            "nav_depth": observations["depth"],  # 用于点导航的深度图
            "robot_xy": robot_xy,  # 机器人XY位置
            "robot_heading": camera_yaw,  # 机器人朝向
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],  # 用于对象地图的RGB-D数据
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],  # 用于价值地图的RGB-D数据
            "habitat_start_yaw": observations["heading"][0].item(),  # 初始朝向角
        }


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    """
    Oracle前端边界探索策略
    使用Oracle提供的边界探索动作
    """
    def _explore(self, observations: TensorDict) -> Tensor:
        """
        使用Oracle提供的边界探索策略进行探索
        
        Args:
            observations: 观察数据
            
        Returns:
            探索动作
        """
        # 查找以_explorer结尾的传感器键
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        # 直接使用该传感器提供的动作
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    """
    超级Oracle前端边界探索策略
    直接使用BaseExplorer提供的动作，跳过其他处理
    """
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        """
        直接使用BaseExplorer提供的动作
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态（未使用）
            
        Returns:
            策略动作数据
        """
        # 直接返回BaseExplorer提供的动作
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    """
    Habitat图像-文本匹配策略
    结合HabitatMixin和ITMPolicy的功能
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    """
    Habitat图像-文本匹配策略V2
    结合HabitatMixin和ITMPolicyV2的功能
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    """
    Habitat图像-文本匹配策略V3
    结合HabitatMixin和ITMPolicyV3的功能
    """
    pass


@dataclass
class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
    """
    视觉语言寻找模型策略配置
    继承VLFMConfig和PolicyConfig，结合两者的配置项
    """
    pass


# 获取配置存储实例
cs = ConfigStore.instance()
# 注册VLFM策略配置，这个配置将用于创建VLFM策略实例，我可以非常快储存我的配置
cs.store(group="habitat_baselines/rl/policy", name="vlfm_policy", node=VLFMPolicyConfig)
