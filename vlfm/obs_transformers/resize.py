# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from gym import spaces
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import (
    ObsTransformConfig,
)
from habitat_baselines.utils.common import (
    get_image_height_width,
    overwrite_gym_box_shape,
)
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from vlfm.obs_transformers.utils import image_resize


@baseline_registry.register_obs_transformer()
class Resize(ObservationTransformer):
    """
    观察变换器：调整观察图像的大小
    用于将模拟环境中的RGB、深度和语义图等观察数据调整为指定大小
    """

    def __init__(
        self,
        size: Tuple[int, int],
        channels_last: bool = True,
        trans_keys: Tuple[str, ...] = ("rgb", "depth", "semantic"),
        semantic_key: str = "semantic",
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension

        参数:
        size: 要调整到的目标尺寸，格式为(高度, 宽度)
        channels_last: 指示通道维度是否在最后，如果为True，则格式为(H,W,C)；否则为(C,H,W)
        trans_keys: 需要调整大小的观察键名元组
        semantic_key: 语义图的键名，用于区分插值方法
        """
        super(Resize, self).__init__()
        self._size: Tuple[int, int] = size  # 存储目标尺寸
        self.channels_last: bool = channels_last  # 存储通道顺序标志
        self.trans_keys: Tuple[str, ...] = trans_keys  # 存储需处理的观察键名
        self.semantic_key = semantic_key  # 存储语义图的键名

    def transform_observation_space(self, observation_space: spaces.Dict) -> spaces.Dict:
        """
        转换观察空间，更新每个需要调整大小的观察的形状信息

        Args:
            observation_space: 原始观察空间

        Returns:
            更新后的观察空间
        """
        observation_space = copy.deepcopy(observation_space)  # 深拷贝以避免修改原始空间
        for key in observation_space.spaces:
            if key in self.trans_keys:
                # 在观察空间字典中，通道总是在最后
                h, w = get_image_height_width(observation_space.spaces[key], channels_last=True)
                if self._size == (h, w):
                    continue  # 如果尺寸已经符合要求，则跳过
                logger.info("Resizing observation of %s: from %s to %s" % (key, (h, w), self._size))
                # 更新观察空间中对应键的形状
                observation_space.spaces[key] = overwrite_gym_box_shape(observation_space.spaces[key], self._size)
        return observation_space

    def _transform_obs(self, obs: torch.Tensor, interpolation_mode: str) -> torch.Tensor:
        """
        调整单个观察张量的大小

        Args:
            obs: 输入观察张量
            interpolation_mode: 插值方式，如'area'或'nearest'

        Returns:
            调整大小后的观察张量
        """
        return image_resize(
            obs,
            self._size,
            channels_last=self.channels_last,
            interpolation_mode=interpolation_mode,
        )

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向处理方法，对所有指定的观察数据进行大小调整

        Args:
            observations: 包含多种观察数据的字典

        Returns:
            处理后的观察数据字典
        """
        for sensor in self.trans_keys:
            if sensor in observations:
                # 默认使用'area'插值方式，更适合普通图像
                interpolation_mode = "area"
                # 对于语义图，使用'nearest'插值以保持类别标签的准确性
                if self.semantic_key in sensor:
                    interpolation_mode = "nearest"
                # 调整观察数据的大小
                observations[sensor] = self._transform_obs(observations[sensor], interpolation_mode)
        return observations

    @classmethod
    def from_config(cls, config: "DictConfig") -> "Resize":
        """
        从配置创建Resize实例的类方法

        Args:
            config: 包含所需参数的配置对象

        Returns:
            配置好的Resize实例
        """
        return cls(
            (int(config.size[0]), int(config.size[1])),  # 确保尺寸为整数元组
            config.channels_last,
            config.trans_keys,
            config.semantic_key,
        )


@dataclass
class ResizeConfig(ObsTransformConfig):
    """
    Resize观察变换器的配置类
    包含初始化Resize所需的所有参数
    """

    type: str = Resize.__name__  # 变换器类型名称
    size: Tuple[int, int] = (224, 224)  # 默认调整为224x224，常用于视觉模型
    channels_last: bool = True  # 默认使用channels_last格式
    trans_keys: Tuple[str, ...] = (  # 默认处理这三种观察数据
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"  # 默认语义图键名


cs = ConfigStore.instance()  # 获取配置存储实例

# 注册配置到配置存储系统中，使其可被Hydra识别和使用
cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.resize",  # 包路径
    group="habitat_baselines/rl/policy/obs_transforms",  # 配置组
    name="resize",  # 配置名
    node=ResizeConfig,  # 配置节点类
)
