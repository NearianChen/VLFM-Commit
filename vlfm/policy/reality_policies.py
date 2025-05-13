# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2

INITIAL_ARM_YAWS = np.deg2rad([-90, -60, -30, 0, 30, 60, 90, 0]).tolist()


class RealityMixin:
    """
    This Python mixin only contains code relevant for running a ITMPolicyV2
    explicitly in the real world (vs. Habitat), and will endow any parent class
    (that is a subclass of ITMPolicyV2) with the necessary methods to run on the
    Spot robot in the real world.
    
    这个Python混入类仅包含在现实世界中运行ITMPolicyV2所需的代码
    （而不是在Habitat模拟器中）。它将为任何父类（ITMPolicyV2的子类）
    提供在现实世界中的Spot机器人上运行所需的方法。
    """

    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # 停止动作张量，线速度和角速度都为0
    _load_yolo: bool = False  # 不加载YOLO模型（真实环境中使用其他检测方法）
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )  # 非COCO数据集物体的描述文本，用于对象检测
    _initial_yaws: List = INITIAL_ARM_YAWS.copy()  # 初始机械臂朝向角度列表
    _observations_cache: Dict[str, Any] = {}  # 观察数据缓存
    _policy_info: Dict[str, Any] = {}  # 策略信息
    _done_initializing: bool = False  # 是否完成初始化标志

    def __init__(self: Union["RealityMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        """
        初始化现实环境混入类
        加载深度预测模型，并设置对象地图参数
        
        Args:
            *args, **kwargs: 传递给父类的参数
        """
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore  # 初始化父类，启用区域同步
        # 加载ZoeDepth深度预测模型，用于从RGB图像估计深度图
        self._depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", config_mode="eval", pretrained=True).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._object_map.use_dbscan = False  # type: ignore  # 禁用DBSCAN聚类，在真实环境中可能造成过度过滤

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        """
        从配置创建策略实例的类方法
        
        Args:
            config: 包含策略配置的配置对象
            
        Returns:
            配置好的RealityMixin实例
        """
        # 获取策略配置部分
        policy_config: VLFMConfig = config.policy
        # 从配置中提取所需参数
        kwargs = {k: policy_config[k] for k in VLFMConfig.kwaarg_names}  # type: ignore

        return cls(**kwargs)

    def act(
        self: Union["RealityMixin", ITMPolicyV2],
        observations: Dict[str, Any],  # 观察数据
        rnn_hidden_states: Union[Tensor, Any],  # RNN隐藏状态
        prev_actions: Any,  # 上一步动作
        masks: Tensor,  # 重置掩码
        deterministic: bool = False,  # 是否确定性行为
    ) -> Dict[str, Any]:
        """
        执行策略动作方法，将模拟环境动作转换为实际机器人动作格式
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 上一步动作
            masks: 重置掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            适用于Spot机器人的动作字典
        """
        # 更新非COCO物体描述，确保包含目标对象
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption
        # 调用父类方法获取动作
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        action: Tensor = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)[0]

        # The output of the policy is a (1, 2) tensor of floats, where the first element
        # is the linear velocity and the second element is the angular velocity. We
        # convert this numpy array to a dictionary with keys "angular" and "linear" so
        # that it can be passed to the Spot robot.
        
        # 策略输出是一个(1,2)形状的浮点张量，第一个元素是线速度，第二个是角速度
        # 将这个张量转换为包含"angular"和"linear"键的字典，便于传递给Spot机器人
        if self._done_initializing:
            # 初始化完成后，正常导航模式
            action_dict = {
                "angular": action[0][0].item(),  # 角速度
                "linear": action[0][1].item(),   # 线速度
                "arm_yaw": -1,  # -1表示不控制机械臂
                "info": self._policy_info,  # 策略信息
            }
        else:
            # 初始化阶段，机器人本体不动，仅旋转机械臂
            action_dict = {
                "angular": 0,  # 不旋转
                "linear": 0,   # 不前进
                "arm_yaw": action[0][0].item(),  # 机械臂偏航角
                "info": self._policy_info,  # 策略信息
            }

        # 添加距离和角度信息（如果有）
        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        # 当所有初始机械臂角度都使用完毕时，认为初始化完成
        self._done_initializing = len(self._initial_yaws) == 0

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        """
        获取动作的简化接口
        
        Args:
            observations: 观察数据
            masks: 重置掩码
            deterministic: 是否确定性，默认为True
            
        Returns:
            动作字典
        """
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["RealityMixin", ITMPolicyV2]) -> None:
        """
        重置策略状态
        恢复初始机械臂角度列表和初始化标志
        """
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        parent_cls._reset()  # 调用父类重置方法
        self._initial_yaws = INITIAL_ARM_YAWS.copy()  # 重置初始角度列表
        self._done_initializing = False  # 重置初始化标志

    def _initialize(self) -> Tensor:
        """
        初始化方法，在每次重置后执行
        逐一返回机械臂的预定义角度序列
        
        Returns:
            下一个机械臂角度
        """
        yaw = self._initial_yaws.pop(0)  # 取出并移除列表第一个角度
        return torch.tensor([[yaw]], dtype=torch.float32)  # 转换为张量返回

    def _cache_observations(self: Union["RealityMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """Caches the rgb, depth, and camera transform from the observations.
        
        缓存观察数据中的RGB图像、深度图像和相机变换等
        处理障碍物地图并更新缓存
        
        Args:
           observations: 当前时间步的观察数据
        """
        # 如果缓存已有数据，则不重复处理
        if len(self._observations_cache) > 0:
            return

        # 更新障碍物地图
        self._obstacle_map: ObstacleMap
        # 处理除最后一个外的所有障碍物地图数据（不更新探索区域）
        for obs_map_data in observations["obstacle_map_depths"][:-1]:
            depth, tf, min_depth, max_depth, fx, fy, topdown_fov = obs_map_data
            self._obstacle_map.update_map(
                depth,
                tf,
                min_depth,
                max_depth,
                fx,
                fy,
                topdown_fov,
                explore=False,  # 不更新探索区域
            )

        # 处理最后一个障碍物地图数据（更新探索区域，但不更新障碍物）
        _, tf, min_depth, max_depth, fx, fy, topdown_fov = observations["obstacle_map_depths"][-1]
        self._obstacle_map.update_map(
            None,  # 不提供深度图，仅更新探索区域
            tf,
            min_depth,
            max_depth,
            fx,
            fy,
            topdown_fov,
            explore=True,  # 更新探索区域
            update_obstacles=False,  # 不更新障碍物
        )

        # 更新代理轨迹并获取边界点
        self._obstacle_map.update_agent_traj(observations["robot_xy"], observations["robot_heading"])
        frontiers = self._obstacle_map.frontiers

        # 处理导航深度图
        height, width = observations["nav_depth"].shape
        nav_depth = torch.from_numpy(observations["nav_depth"])
        nav_depth = nav_depth.reshape(1, height, width, 1).to("cuda")  # 重塑并移至GPU

        # 构建观察缓存
        self._observations_cache = {
            "frontier_sensor": frontiers,  # 边界点
            "nav_depth": nav_depth,  # 用于点导航的深度图
            "robot_xy": observations["robot_xy"],  # (2,) np.ndarray，机器人位置
            "robot_heading": observations["robot_heading"],  # float，机器人朝向（弧度）
            "object_map_rgbd": observations["object_map_rgbd"],  # 用于对象地图的RGB-D数据
            "value_map_rgbd": observations["value_map_rgbd"],  # 用于价值地图的RGB-D数据
        }

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.
        
        从RGB图像推断深度图像
        在深度传感器数据不可用时使用
        
        Args:
            rgb: 输入RGB图像
            min_depth: 最小深度值
            max_depth: 最大深度值
            
        Returns:
            推断的归一化深度图像
        """
        # 将NumPy数组转换为PIL图像
        img_pil = Image.fromarray(rgb)
        # 使用ZoeDepth模型推断深度，禁用梯度计算
        with torch.inference_mode():
            depth = self._depth_model.infer_pil(img_pil)
        # 裁剪深度值并归一化到[0,1]范围
        depth = (np.clip(depth, min_depth, max_depth)) / (max_depth - min_depth)
        return depth


@dataclass
class RealityConfig(DictConfig):
    """
    现实环境配置类
    包含用于现实环境中的策略配置
    """
    policy: VLFMConfig = VLFMConfig()  # 使用默认VLFM配置


class RealityITMPolicyV2(RealityMixin, ITMPolicyV2):
    """
    现实环境中的图像-文本匹配策略V2
    结合RealityMixin和ITMPolicyV2的功能
    适用于实际机器人系统
    """
    pass
