# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Generator

import torch
from habitat import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData


@baseline_registry.register_policy  # 注册此策略到Habitat的策略注册表中
class BasePolicy(Policy):
    """The bare minimum needed to load a policy for evaluation using ppo_trainer.py
    
    这是一个最小化的基础策略类，提供了使用Habitat的PPO训练器进行评估所需的最基本功能。
    作为其他复杂策略类的基类，定义了与Habitat框架集成所需的基本接口。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化基础策略
        接受任意参数以兼容不同的子类初始化需求
        """
        super().__init__()

    @property
    def should_load_agent_state(self) -> bool:
        """
        指示是否应加载代理状态
        
        Returns:
            False表示不从检查点加载代理状态
        """
        return False

    @classmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> Any:
        """
        从配置创建策略实例的类方法
        
        Returns:
            策略实例
        """
        return cls()

    def act(
        self,
        observations: TensorDict,  # 观察数据字典
        rnn_hidden_states: torch.Tensor,  # RNN隐藏状态
        prev_actions: torch.Tensor,  # 前一步动作
        masks: torch.Tensor,  # 重置掩码
        deterministic: bool = False,  # 是否使用确定性策略
    ) -> PolicyActionData:
        """
        根据当前观察决定动作
        这个基本实现只是简单地向前移动（总是执行动作1）
        
        Args:
            observations: 环境观察数据
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 上一步执行的动作
            masks: 用于RNN的掩码，表示哪些环境已重置
            deterministic: 是否使用确定性策略
            
        Returns:
            包含动作和RNN隐藏状态的策略动作数据
        """
        # 只是向前移动
        num_envs = observations["rgb"].shape[0]  # 获取环境数量
        action = torch.ones(num_envs, 1, dtype=torch.long)  # 创建全1动作张量
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    # used in ppo_trainer.py eval:

    def to(self, *args: Any, **kwargs: Any) -> None:
        """
        将模型移动到指定设备的方法
        这里是空实现，因为基础策略没有可训练参数需要移动
        """
        return

    def eval(self) -> None:
        """
        将模型设置为评估模式
        这里是空实现，因为基础策略没有需要切换模式的组件
        """
        return

    def parameters(self) -> Generator:
        """
        获取模型参数的生成器
        这里只生成一个零张量，用于兼容优化器接口
        
        Returns:
            参数生成器，只包含一个零张量
        """
        yield torch.zeros(1)


if __name__ == "__main__":
    """
    主程序入口
    创建并保存一个虚拟模型检查点，用于加载不需要实际检查点的策略
    即使策略不读取检查点数据，Habitat仍然要求加载一个检查点文件
    """
    # 保存一个虚拟的state_dict使用torch.save
    # 这对于生成一个可用于加载其他不需要读取检查点的策略的pth文件很有用
    # 尽管Habitat要求加载一个检查点
    config = get_config("habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml")
    dummy_dict = {
        "config": config,  # 配置对象
        "extra_state": {"step": 0},  # 额外状态信息
        "state_dict": {},  # 空模型状态字典
    }

    torch.save(dummy_dict, "dummy_policy.pth")  # 保存为dummy_policy.pth文件
