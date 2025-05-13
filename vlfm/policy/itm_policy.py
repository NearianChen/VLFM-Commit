# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"  # 文本提示分隔符，用于分隔多个文本提示


class BaseITMPolicy(BaseObjectNavPolicy):
    """
    基础图像-文本匹配策略类
    使用图像-文本相似度来指导机器人探索和导航到目标对象
    """
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)  # 目标对象标记颜色（绿色）
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)  # 选中的边界点颜色（黄色）
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)  # 边界点颜色（红色）
    _circle_marker_thickness: int = 2  # 圆形标记线条粗细
    _circle_marker_radius: int = 5  # 圆形标记半径
    _last_value: float = float("-inf")  # 上一个边界点的价值，初始为负无穷
    _last_frontier: np.ndarray = np.zeros(2)  # 上一个选择的边界点坐标

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        """
        可视化时用于降维多通道价值地图的函数
        默认取每个位置所有通道的最大值
        
        Args:
            i: 多通道价值地图数组
            
        Returns:
            单通道价值地图（取最大值）
        """
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,  # 文本提示，用于计算图像-文本相似度
        use_max_confidence: bool = True,  # 是否使用最大置信度
        sync_explored_areas: bool = False,  # 是否同步已探索区域
        *args: Any,
        **kwargs: Any,
    ):
        """
        初始化图像-文本匹配策略
        
        Args:
            text_prompt: 文本提示模板，其中"target_object"将被替换为实际目标
            use_max_confidence: 是否使用最大置信度而非加权平均
            sync_explored_areas: 是否将价值地图与障碍物地图的已探索区域同步
        """
        super().__init__(*args, **kwargs)
        # 初始化BLIP2图像-文本匹配客户端
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        # 初始化价值地图
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),  # 根据提示数量确定通道数
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        # 初始化循环避免器，防止机器人在同一区域循环
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        """
        重置策略状态
        清空价值地图、循环避免器和历史记录
        """
        super()._reset()  # 调用父类的重置方法
        self._value_map.reset()  # 重置价值地图
        self._acyclic_enforcer = AcyclicEnforcer()  # 重置循环避免器
        self._last_value = float("-inf")  # 重置上一个价值
        self._last_frontier = np.zeros(2)  # 重置上一个边界点

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        """
        探索策略实现
        根据当前边界点和它们的价值选择最佳探索方向
        
        Args:
            observations: 观察数据
            
        Returns:
            探索动作
        """
        # 获取可用的边界点
        frontiers = self._observations_cache["frontier_sensor"]
        # 如果没有可用的边界点，则停止
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        # 获取最佳边界点及其价值
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        # 记录调试信息
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        # 使用点导航策略移动到最佳边界点
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.
        
        根据价值地图选择最佳边界点及其价值
        考虑持续性（倾向于继续探索上一个选择的边界点）和避免循环

        Args:
            observations: 环境观察数据
            frontiers: 待选择的边界点数组，2D点的数组

        Returns:
            Tuple[np.ndarray, float]: 最佳边界点及其价值
        """
        # 对边界点按价值降序排序
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]  # 获取机器人当前位置
        best_frontier_idx = None  # 最佳边界点索引
        top_two_values = tuple(sorted_values[:2])  # 记录前两个最高价值，用于循环检测

        os.environ["DEBUG_INFO"] = ""
        # 如果上一次有选择的边界点，考虑继续使用该点（持续性）
        # 前提是该点仍在边界列表中且价值没有显著下降
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            # 检查上一个边界点是否仍在列表中
            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # 上一个边界点仍在边界列表中
                    curr_index = idx
                    break

            # 如果上一个点不在列表中，查找附近的点
            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # 存在接近上一个边界点的点
                    curr_index = closest_index

            # 如果找到了上一个点或其附近的点
            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                # 如果当前价值没有明显变差（不超过0.01）
                if curr_value + 0.01 > self._last_value:
                    # 坚持使用上一个边界点或其附近点
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # 如果没有继续使用上一个边界点（或附近点），选择最佳点，前提是不会造成循环
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                # 检查该边界点是否会导致循环路径
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        # 如果所有点都会导致循环，则选择离机器人最远的点
        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        # 获取最佳边界点及其价值
        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        # 更新循环避免器状态
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        # 记录选择的边界点和价值
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """
        获取策略信息，用于可视化和记录
        添加价值地图和边界点标记到策略信息中
        
        Args:
            detections: 对象检测结果
            
        Returns:
            策略信息字典
        """
        # 获取基类的策略信息
        policy_info = super()._get_policy_info(detections)

        # 如果不需要可视化，直接返回
        if not self._visualize:
            return policy_info

        markers = []  # 存储要在地图上绘制的标记

        # 在代价地图上绘制边界点
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        # 如果有目标位置，在代价地图上绘制
        if not np.array_equal(self._last_goal, np.zeros(2)):
            # 判断目标是否为边界点，决定使用哪种颜色
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color  # 选中的边界点颜色
            else:
                color = self._target_object_color  # 目标对象颜色
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
            
        # 添加价值地图可视化
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,  # 转换为RGB格式（OpenCV默认使用BGR）
        )

        return policy_info

    def _update_value_map(self) -> None:
        """
        更新价值地图
        计算当前视图与提示文本的图像-文本匹配分数
        更新价值地图和代理轨迹
        """
        # 获取所有RGB图像
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        # 计算每个图像与每个文本提示的余弦相似度
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),  # 替换目标对象名称
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)  # 处理每个文本提示
            ]
            for rgb in all_rgb
        ]
        # 更新价值地图
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        # 更新代理轨迹
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        根据价值对边界点进行排序
        这是一个抽象方法，需要子类实现
        
        Args:
            observations: 观察数据
            frontiers: 边界点数组
            
        Returns:
            排序后的边界点和对应的价值列表
        """
        raise NotImplementedError  # 子类必须实现此方法


class ITMPolicy(BaseITMPolicy):
    """
    图像-文本匹配策略（第一版）
    使用边界地图（FrontierMap）来管理和评估边界点
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化ITM策略
        创建边界地图用于边界点管理
        """
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()  # 创建边界地图

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        行为决策方法
        预处理观察数据，更新价值地图，然后决定行为
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步动作
            masks: 重置掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            动作张量和RNN隐藏状态元组
        """
        self._pre_step(observations, masks)  # 预处理观察数据
        if self._visualize:  # 如果需要可视化，更新价值地图
            self._update_value_map()
        # 调用父类的act方法决定行为
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        """
        重置策略状态
        清空边界地图和其他状态
        """
        super()._reset()
        self._frontier_map.reset()  # 重置边界地图

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        使用边界地图对边界点进行排序
        基于当前图像与目标描述的相似度
        
        Args:
            observations: 观察数据
            frontiers: 边界点数组
            
        Returns:
            排序后的边界点和价值列表
        """
        # 获取RGB图像
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        # 替换文本提示中的target_object为实际目标名称
        text = self._text_prompt.replace("target_object", self._target_object)
        # 更新边界地图
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        # 获取排序后的边界点和价值
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    """
    图像-文本匹配策略（第二版）
    直接使用价值地图对边界点进行评估和排序
    比第一版更直接地利用价值地图信息
    """
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        行为决策方法
        每次都更新价值地图，然后决定行为
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步动作
            masks: 重置掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            决策结果
        """
        self._pre_step(observations, masks)
        self._update_value_map()  # 总是更新价值地图
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        使用价值地图对边界点进行排序
        根据点周围0.5米半径内的价值评估每个边界点
        
        Args:
            observations: 观察数据
            frontiers: 边界点数组
            
        Returns:
            排序后的边界点和价值列表
        """
        # 使用价值地图的sort_waypoints方法对边界点进行排序
        # 0.5表示评估每个点周围0.5米半径内的价值
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    """
    图像-文本匹配策略（第三版）
    在V2基础上增加了探索阈值
    当最高目标价值低于阈值时，转向探索模式而非目标导向模式
    """
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        """
        初始化ITMPolicyV3
        
        Args:
            exploration_thresh: 探索阈值，当目标价值低于此值时转为探索模式
        """
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh  # 探索阈值

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            """
            针对双通道价值地图的可视化函数
            根据阈值决定使用目标通道或探索通道的值
            
            Args:
                arr: 多通道价值地图数组
                
            Returns:
                可视化后的单通道数组
            """
            # 获取第一个通道（目标通道）的值
            first_channel = arr[:, :, 0]
            # 获取所有通道的最大值
            max_values = np.max(arr, axis=2)
            # 创建布尔掩码，标记第一通道值高于阈值的位置
            mask = first_channel > exploration_thresh
            # 根据掩码选择使用第一通道或最大值
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore  # 覆盖可视化函数

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        使用改进的价值排序方法
        根据探索阈值决定使用目标价值或探索价值
        
        Args:
            observations: 观察数据
            frontiers: 边界点数组
            
        Returns:
            排序后的边界点和价值列表
        """
        # 调用价值地图的sort_waypoints方法，使用_reduce_values函数处理多通道价值
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier
        
        将每个边界点的多通道价值降维为单一价值
        根据最高目标价值是否超过阈值决定使用目标价值还是探索价值
        
        Args:
            values: 价值元组列表，形式为(目标价值,探索价值)
                   如果所有目标价值的最大值低于阈值，则返回每个元组的第二个元素（探索价值）
                   否则返回每个元组的第一个元素（目标价值）
                   
        Returns:
            降维后的价值列表，每个边界点一个价值
        """
        # 提取所有目标价值（每个元组的第一个元素）
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)  # 获取最高目标价值

        # 如果最高目标价值低于探索阈值，使用探索价值
        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]  # 提取每个元组的第二个元素（探索价值）
            return explore_values
        else:
            # 否则使用目标价值
            return [v[0] for v in values]  # 提取每个元组的第一个元素（目标价值）
