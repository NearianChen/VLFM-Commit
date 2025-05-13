# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    """
    基础目标导航策略类
    整合视觉语言模型和点云地图，实现目标检测和导航功能
    该类为抽象基类，需要子类实现特定方法
    """
    _target_object: str = ""  # 目标对象名称
    _policy_info: Dict[str, Any] = {}  # 策略信息字典，用于记录状态和可视化
    _object_masks: Union[np.ndarray, Any] = None  # 对象掩码，由_update_object_map()设置
    _stop_action: Union[Tensor, Any] = None  # 停止动作，必须由子类设置
    _observations_cache: Dict[str, Any] = {}  # 观察缓存
    _non_coco_caption = ""  # 非COCO类别的描述提示
    _load_yolo: bool = True  # 是否加载YOLOv7模型

    def __init__(
        self,
        pointnav_policy_path: str,  # 点导航策略权重路径
        depth_image_shape: Tuple[int, int],  # 深度图像形状
        pointnav_stop_radius: float,  # 点导航停止半径
        object_map_erosion_size: float,  # 对象地图腐蚀大小
        visualize: bool = True,  # 是否可视化
        compute_frontiers: bool = True,  # 是否计算边界
        min_obstacle_height: float = 0.15,  # 最小障碍物高度
        max_obstacle_height: float = 0.88,  # 最大障碍物高度
        agent_radius: float = 0.18,  # 代理半径
        obstacle_map_area_threshold: float = 1.5,  # 障碍地图区域阈值
        hole_area_thresh: int = 100000,  # 洞区域阈值
        use_vqa: bool = False,  # 是否使用视觉问答
        vqa_prompt: str = "Is this ",  # 视觉问答提示
        coco_threshold: float = 0.8,  # COCO类别置信度阈值
        non_coco_threshold: float = 0.4,  # 非COCO类别置信度阈值
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        初始化基础目标导航策略
        
        Args:
            pointnav_policy_path: 预训练点导航策略权重文件路径
            depth_image_shape: 处理深度图像的目标形状
            pointnav_stop_radius: 到达目标时的停止半径（米）
            object_map_erosion_size: 对象地图腐蚀大小，影响分割质量
            visualize: 是否生成可视化数据
            compute_frontiers: 是否计算和使用探索边界
            min_obstacle_height: 最小障碍物高度，低于此高度的物体不被视为障碍
            max_obstacle_height: 最大障碍物高度，高于此高度的物体不被视为障碍
            agent_radius: 代理机器人的半径，影响导航路径规划
            obstacle_map_area_threshold: 障碍物地图区域阈值（平方米）
            hole_area_thresh: 要填充的深度图中洞的面积阈值
            use_vqa: 是否使用BLIP2视觉问答进行对象验证
            vqa_prompt: 视觉问答的提示前缀
            coco_threshold: COCO类别对象检测的置信度阈值
            non_coco_threshold: 非COCO类别对象检测的置信度阈值
        """
        super().__init__()
        # 初始化各种对象检测和分割模型客户端
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        # 初始化点导航策略
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        # 初始化对象点云地图
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        # 状态跟踪变量
        self._num_steps = 0  # 步数计数器
        self._did_reset = False  # 是否已重置
        self._last_goal = np.zeros(2)  # 上一个目标位置
        self._done_initializing = False  # 是否完成初始化
        self._called_stop = False  # 是否已调用停止
        self._compute_frontiers = compute_frontiers  # 是否计算边界
        if compute_frontiers:
            # 初始化障碍物地图，用于边界探索
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def _reset(self) -> None:
        """
        重置策略状态
        在每个新的导航任务开始时调用
        """
        self._target_object = ""  # 清空目标对象
        self._pointnav_policy.reset()  # 重置点导航策略
        self._object_map.reset()  # 重置对象地图
        self._last_goal = np.zeros(2)  # 重置上一个目标
        self._num_steps = 0  # 重置步数计数器
        self._done_initializing = False  # 重置初始化标志
        self._called_stop = False  # 重置停止标志
        if self._compute_frontiers:
            self._obstacle_map.reset()  # 重置障碍物地图
        self._did_reset = True  # 设置重置标志

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        
        行动决策的主要方法，根据当前状态决定下一步动作
        行为分三个阶段：
        1. 初始化阶段：机器人原地旋转，获取环境信息
        2. 探索阶段：如果未找到目标对象，则探索环境
        3. 导航阶段：一旦找到目标对象，则向其导航
        
        Args:
            observations: 当前环境观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步动作
            masks: 重置掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            动作张量和RNN隐藏状态元组
        """
        # 前处理步骤，处理观察并更新目标对象
        self._pre_step(observations, masks)

        # 获取对象地图所需的RGB-D数据
        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        # 更新对象地图并获取检测结果
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        # 获取机器人当前位置
        robot_xy = self._observations_cache["robot_xy"]
        # 获取目标对象位置
        goal = self._get_target_object_location(robot_xy)

        # 根据当前状态决定行为模式和动作
        if not self._done_initializing:  # 初始化阶段
            mode = "initialize"
            pointnav_action = self._initialize()  # 子类实现
        elif goal is None:  # 探索阶段：未找到目标对象
            mode = "explore"
            pointnav_action = self._explore(observations)  # 子类实现
        else:  # 导航阶段：已找到目标对象，前往目标位置
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        # 打印当前步数、模式和动作信息
        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        # 更新策略信息
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

        # 清空观察缓存，重置标志
        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        """
        执行动作前的准备工作
        处理重置信号，设置目标对象，缓存观察结果
        
        Args:
            observations: 观察数据
            masks: 重置掩码，0表示应该重置
        """
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        # 如果收到重置信号且未重置，则进行重置
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]  # 设置目标对象
        try:
            # 缓存当前观察数据
            self._cache_observations(observations)  # 子类实现
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration  # 到达地图边缘，停止迭代
        self._policy_info = {}  # 清空策略信息

    def _initialize(self) -> Tensor:
        """
        初始化阶段的行为策略
        通常是让机器人原地旋转以获取环境信息
        
        Returns:
            初始化阶段的动作张量
        """
        raise NotImplementedError  # 子类必须实现

    def _explore(self, observations: "TensorDict") -> Tensor:
        """
        探索阶段的行为策略
        当尚未发现目标对象时，探索环境
        
        Args:
            observations: 当前观察数据
            
        Returns:
            探索阶段的动作张量
        """
        raise NotImplementedError  # 子类必须实现

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        获取目标对象的位置
        如果尚未发现目标对象，则返回None
        
        Args:
            position: 当前机器人位置
            
        Returns:
            目标对象的位置或None
        """
        if self._object_map.has_object(self._target_object):
            # 如果对象地图中有目标对象，获取最佳对象位置
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None  # 未找到目标对象

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """
        获取策略信息，用于可视化和记录
        
        Args:
            detections: 对象检测结果
            
        Returns:
            包含策略信息的字典
        """
        # 获取目标对象的点云数据（如果有）
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
            
        # 构建基本策略信息
        policy_info = {
            "target_object": self._target_object.split("|")[0],  # 目标对象名称
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),  # GPS位置
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),  # 朝向角度
            "target_detected": self._object_map.has_object(self._target_object),  # 是否检测到目标
            "target_point_cloud": target_point_cloud,  # 目标点云
            "nav_goal": self._last_goal,  # 当前导航目标
            "stop_called": self._called_stop,  # 是否已调用停止
            # 不在图像上渲染的项目：
            "render_below_images": [
                "target_object",
            ],
        }

        # 如果不需要可视化，直接返回
        if not self._visualize:
            return policy_info

        # 处理深度图可视化
        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # 如果有对象掩码，绘制轮廓
        if self._object_masks.sum() > 0:
            # 查找并绘制对象轮廓
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
            
        # 添加可视化图像到策略信息
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # 如果启用了边界计算，添加障碍物地图可视化
        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        # 如果有调试信息，添加到策略信息
        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        """
        获取图像中的对象检测结果
        根据目标对象类别，使用合适的检测器
        
        Args:
            img: 输入RGB图像
            
        Returns:
            对象检测结果
        """
        # 解析目标类别，可能有多个类别用|分隔
        target_classes = self._target_object.split("|")
        # 判断目标是否为COCO类别
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        # 根据目标类型选择检测器
        detections = (
            self._coco_object_detector.predict(img)  # 使用YOLOv7检测COCO类别
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)  # 使用DINO检测非COCO类别
        )
        # 根据目标类别过滤检测结果
        detections.filter_by_class(target_classes)
        # 根据目标类型设置置信度阈值并过滤
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        # 如果同时有COCO和非COCO类别，且未检测到，尝试使用DINO检测器
        if has_coco and has_non_coco and detections.num_detections == 0:
            # 重试使用非COCO对象检测器
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.
        
        使用预训练的点导航策略向目标位置移动
        计算机器人当前位置到目标位置的距离和角度，然后确定下一步动作
        
        Args:
            goal: 目标位置坐标，格式为(x, y)，单位为米
            stop: 如果接近目标是否停止
            
        Returns:
            点导航动作张量
        """
        # 创建掩码张量，指示是否为首次调用
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        
        # 如果目标位置发生显著变化，重置点导航策略
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:  # 目标变化超过10厘米
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)  # 重置掩码
            self._last_goal = goal  # 更新上一个目标
            
        # 获取机器人当前位置和朝向
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        
        # 计算到目标的距离和角度
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        
        # 构建点导航观察数据
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        
        # 记录距离和角度信息
        self._policy_info["rho_theta"] = np.array([rho, theta])
        
        # 如果接近目标且允许停止，则停止
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
            
        # 使用点导航策略决定动作
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.
        
        使用RGB和深度图像更新对象地图
        检测目标对象，创建分割掩码，更新3D点云地图
        
        Args:
            rgb: RGB图像，用于对象检测和分割
            depth: 深度图像，归一化到[0,1]范围
            tf_camera_to_episodic: 从相机到全局坐标系的变换矩阵
            min_depth: 深度图像的最小深度值（米）
            max_depth: 深度图像的最大深度值（米）
            fx: 相机X方向焦距
            fy: 相机Y方向焦距
            
        Returns:
            对象检测结果
        """
        # 获取对象检测结果
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        # 创建空对象掩码
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        
        # 如果深度图全为1（无效深度）且检测到对象，则推断深度
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)  # 子类实现
            # 更新缓存中的深度图
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
            
        # 处理每个检测到的对象
        for idx in range(len(detections.logits)):
            # 将归一化的边界框坐标转换回像素坐标
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            # 使用SAM对检测到的对象进行分割
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # 如果启用视觉问答，使用BLIP2模型验证分割结果
            if self._use_vqa:
                # 绘制分割轮廓
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                # 构建视觉问答问题
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "  # 添加冠词
                question += detections.phrases[idx] + "? Answer:"
                # 调用视觉问答模型
                answer = self._vqa.ask(annotated_rgb, question)
                # 如果回答不是肯定的，跳过此检测
                if not answer.lower().startswith("yes"):
                    continue

            # 更新对象掩码
            self._object_masks[object_mask > 0] = 1
            # 更新对象点云地图
            self._object_map.update_map(
                self._target_object,  # 目标对象名称
                depth,               # 深度图
                object_mask,         # 对象分割掩码
                tf_camera_to_episodic,  # 坐标变换矩阵
                min_depth,           # 最小深度
                max_depth,           # 最大深度
                fx, fy,              # 相机参数
            )

        # 计算相机视场角
        cone_fov = get_fov(fx, depth.shape[1])
        # 更新已探索区域
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.
        
        从观察数据中提取RGB图像、深度图像和相机变换信息
        缓存处理后的观察数据
        
        Args:
            observations: 当前时间步的观察数据
        """
        raise NotImplementedError  # 子类必须实现

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.
        
        从RGB图像推断深度图像
        在深度传感器数据不可用或无效时使用
        
        Args:
            rgb: 输入RGB图像
            min_depth: 最小深度值
            max_depth: 最大深度值
            
        Returns:
            推断的深度图像
        """
        raise NotImplementedError  # 子类必须实现


@dataclass
class VLFMConfig:
    """
    视觉语言寻找模型(VLFM)的配置类
    包含导航策略所需的各种参数
    """
    name: str = "HabitatITMPolicy"  # 策略名称
    text_prompt: str = "Seems like there is a target_object ahead."  # 文本提示
    pointnav_policy_path: str = "data/pointnav_weights.pth"  # 点导航策略路径
    depth_image_shape: Tuple[int, int] = (224, 224)  # 深度图像形状
    pointnav_stop_radius: float = 0.9  # 点导航停止半径（米）
    use_max_confidence: bool = False  # 是否使用最大置信度
    object_map_erosion_size: int = 5  # 对象地图腐蚀大小
    exploration_thresh: float = 0.0  # 探索阈值
    obstacle_map_area_threshold: float = 1.5  # 障碍地图面积阈值（平方米）
    min_obstacle_height: float = 0.61  # 最小障碍物高度
    max_obstacle_height: float = 0.88  # 最大障碍物高度
    hole_area_thresh: int = 100000  # 洞区域阈值
    use_vqa: bool = False  # 是否使用视觉问答
    vqa_prompt: str = "Is this "  # 视觉问答提示
    coco_threshold: float = 0.8  # COCO类别阈值
    non_coco_threshold: float = 0.4  # 非COCO类别阈值
    agent_radius: float = 0.18  # 代理半径

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        """
        返回所有配置字段名称，除了name字段
        用于自动构建关键字参数
        """
        # 返回除"name"外的所有字段名
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


# 获取配置存储实例
cs = ConfigStore.instance()
# 注册基础VLFM配置
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
