# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, List

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure#这里是注册测量器的装饰器，主要被用来将自定义测量器注册到Habitat的测量器注册表中
class TraveledStairs(Measure):
    """
    楼梯爬行测量类
    用于检测代理是否在模拟中爬上或下了楼梯
    通过监测代理垂直位置(z轴)的变化来确定
    """

    cls_uuid: str = "traveled_stairs"  # 测量类的唯一标识符

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        """
        初始化楼梯爬行测量器

        Args:
            sim: Habitat模拟器实例
            config: 测量配置
            *args, **kwargs: 额外参数
        """
        self._sim = sim  # 存储模拟器引用
        self._config = config  # 存储配置
        self._history: List[np.ndarray] = []  # 初始化空的垂直位置历史记录列表
        super().__init__(*args, **kwargs)  # 调用父类初始化方法

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        """
        获取此测量类的唯一标识符

        Returns:
            测量类的UUID
        """
        return TraveledStairs.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        """
        重置测量指标
        清空位置历史记录并更新当前指标值
        """
        self._history = []  # 清空历史记录
        self.update_metric()  # 更新指标值

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        """
        更新测量指标
        记录当前垂直位置并判断是否达到爬楼梯标准
        """
        curr_z = self._sim.get_agent_state().position[1]  # 获取代理当前的垂直位置(y坐标)
        self._history.append(curr_z)  # 将当前位置添加到历史记录
        # 若历史垂直位置的峰峰值(最大值减最小值)大于0.9米，则认为代理已经爬了楼梯
        # np.ptp计算数组中的峰峰值(peak to peak)
        # int()将布尔值转换为0或1
        self._metric = int(np.ptp(self._history) > 0.9)


@dataclass
class TraveledStairsMeasurementConfig(MeasurementConfig):
    """
    楼梯爬行测量配置类
    继承自Habitat的MeasurementConfig
    """

    type: str = TraveledStairs.__name__  # 配置类型名，设为测量类的名称


# 获取配置存储实例
cs = ConfigStore.instance()
# 将配置注册到Habitat的配置系统中
cs.store(
    package="habitat.task.measurements.traveled_stairs",  # 包路径
    group="habitat/task/measurements",  # 配置组
    name="traveled_stairs",  # 配置名称
    node=TraveledStairsMeasurementConfig,  # 配置节点类
)
