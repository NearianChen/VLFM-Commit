# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from torch import Tensor

# 记录habitat的版本
habitat_version = ""

try:
    # 尝试导入habitat库以使用其官方实现的PointNavResNetPolicy
    import habitat
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    habitat_version = habitat.__version__

    if habitat_version == "0.1.5":
        print("Using habitat 0.1.5; assuming SemExp code is being used")

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):
            """
            适用于habitat 0.1.5版本的策略封装类
            简化了act方法的返回值，只返回动作和RNN隐藏状态
            """
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                value, action, action_log_probs, rnn_hidden_states = super().act(*args, **kwargs)
                return action, rnn_hidden_states

    else:
        # 适配更新版本的habitat
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
            """
            适用于较新habitat版本的策略封装类
            从PolicyActionData中提取动作和RNN隐藏状态
            """
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                policy_actions: "PolicyActionData" = super().act(*args, **kwargs)
                return policy_actions.actions, policy_actions.rnn_hidden_states

    HABITAT_BASELINES_AVAILABLE = True
except ModuleNotFoundError:
    # 如果habitat库不可用，使用本地实现的PointNavResNetPolicy
    from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy,
    )

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
        """Already outputs a tensor, so no need to convert.
        
        已经输出张量格式，无需转换
        """
        pass

    HABITAT_BASELINES_AVAILABLE = False


class WrappedPointNavResNetPolicy:
    """
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    
    PointNavResNetPolicy的封装类，使其更易于使用
    但只能同时处理一个环境。自动更新隐藏状态和前一个动作。
    """

    def __init__(
        self,
        ckpt_path: str,  # 检查点文件路径
        device: Union[str, torch.device] = "cuda",  # 运行设备
    ):
        """
        初始化封装策略
        
        Args:
            ckpt_path: 预训练模型检查点路径
            device: 运行设备，默认为CUDA
        """
        if isinstance(device, str):
            device = torch.device(device)  # 将字符串设备名转换为torch.device对象
        self.policy = load_pointnav_policy(ckpt_path)  # 加载点导航策略
        self.policy.to(device)  # 将策略移至指定设备
        # 判断是否为离散动作空间
        discrete_actions = not hasattr(self.policy.action_distribution, "mu_maybe_std")
        # 初始化RNN隐藏状态
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # 环境数量
            self.policy.net.num_recurrent_layers,  # RNN层数
            512,  # 隐藏状态大小
            device=device,
        )
        # 根据动作类型初始化前一动作张量
        if discrete_actions:
            num_actions = 1
            action_dtype = torch.long  # 离散动作使用长整型
        else:
            num_actions = 2  # 连续动作[线速度, 角速度]
            action_dtype = torch.float32  # 连续动作使用浮点型
        self.pointnav_prev_actions = torch.zeros(
            1,  # 环境数量
            num_actions,
            device=device,
            dtype=action_dtype,
        )
        self.device = device

    def act(
        self,
        observations: Union["TensorDict", Dict],  # 观察数据
        masks: Tensor,  # 重置掩码
        deterministic: bool = False,  # 是否使用确定性策略
    ) -> Tensor:
        """Infers action to take towards the given (rho, theta) based on depth vision.
        
        基于深度视觉推断要采取的动作，以到达给定的(rho, theta)目标位置

        Args:
            observations: 包含至少以下内容的字典:
                    - "depth": 深度图像张量 (N, H, W, 1)
                    - "pointgoal_with_gps_compass": 
                        表示相对于代理当前姿态的距离和角度(rho, theta)的张量 (N, 2)
            masks: 掩码张量，对于一个回合中第一步之后的任何步骤值为1；第一步为0
            deterministic: 是否确定性选择动作（使用策略的均值而非采样）

        Returns:
            表示要采取的动作的张量
        """
        # 将NumPy数组转换为torch张量，并移至目标设备
        observations = move_obs_to_device(observations, self.device)
        # 调用策略的act方法获取动作和RNN隐藏状态
        pointnav_action, rnn_hidden_states = self.policy.act(
            observations,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )
        # 更新前一动作和RNN隐藏状态，用于下一步决策
        self.pointnav_prev_actions = pointnav_action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states
        return pointnav_action

    def reset(self) -> None:
        """
        重置策略的隐藏状态和前一动作
        用于新回合开始时
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(self.pointnav_test_recurrent_hidden_states)
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def load_pointnav_policy(file_path: str) -> PointNavResNetTensorOutputPolicy:
    """Loads a PointNavResNetPolicy policy from a .pth file.
    
    从.pth文件加载PointNavResNetPolicy策略

    Args:
        file_path: 包含预训练策略权重的文件路径
        
    Returns:
        加载好的PointNavResNetTensorOutputPolicy策略
    """
    if HABITAT_BASELINES_AVAILABLE:
        # 如果habitat库可用，构建观察空间和动作空间
        obs_space = SpaceDict(
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),  # 深度图空间
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),  # 目标位置空间
            }
        )
        action_space = Discrete(4)  # 离散动作空间，4个动作
        if habitat_version == "0.1.5":
            # 对于0.1.5版本，使用特定参数初始化策略
            pointnav_policy = PointNavResNetTensorOutputPolicy(
                obs_space,
                action_space,
                hidden_size=512,
                num_recurrent_layers=2,
                rnn_type="LSTM",
                resnet_baseplanes=32,
                backbone="resnet18",
                normalize_visual_inputs=False,
                obs_transform=None,
            )
            # 需要重写视觉编码器，因为它使用的ResNet版本计算压缩大小的方式不同
            from vlfm.policy.utils.non_habitat_policy import (
                PointNavResNetNet,
            )

            # print(pointnav_policy)
            pointnav_policy.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
            # 使用 weights_only=False 以兼容 PyTorch 2.6+ 的安全限制
            state_dict = torch.load(file_path + ".state_dict", map_location="cpu", weights_only=False)
        else:
            # 对于较新版本，使用from_config方法初始化
            try:
                # 首先尝试显式设置 weights_only=False 以处理 PyTorch 2.6+ 安全限制
                ckpt_dict = torch.load(file_path, map_location="cpu", weights_only=False)
            except TypeError:
                # 对于较早版本的 PyTorch，weights_only 参数可能不存在
                ckpt_dict = torch.load(file_path, map_location="cpu")
            pointnav_policy = PointNavResNetTensorOutputPolicy.from_config(ckpt_dict["config"], obs_space, action_space)
            state_dict = ckpt_dict["state_dict"]
        # 加载模型权重
        pointnav_policy.load_state_dict(state_dict)
        return pointnav_policy

    else:
        # 如果habitat不可用，使用本地实现
        ckpt_dict = torch.load(file_path, map_location="cpu")
        pointnav_policy = PointNavResNetTensorOutputPolicy()
        current_state_dict = pointnav_policy.state_dict()
        # 让旧检查点能够与新代码一起工作（处理键名变化）
        if "net.prev_action_embedding_cont.bias" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.bias"] = ckpt_dict["net.prev_action_embedding.bias"]
        if "net.prev_action_embedding_cont.weights" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.weight"] = ckpt_dict["net.prev_action_embedding.weight"]

        # 只加载当前模型中存在的参数
        pointnav_policy.load_state_dict({k: v for k, v in ckpt_dict.items() if k in current_state_dict})
        # 打印未加载的键
        unused_keys = [k for k in ckpt_dict.keys() if k not in current_state_dict]
        print(f"The following unused keys were not loaded when loading the pointnav policy: {unused_keys}")
        return pointnav_policy


def move_obs_to_device(
    observations: Dict[str, Any],
    device: torch.device,
    unsqueeze: bool = False,
) -> Dict[str, Tensor]:
    """Moves observations to the given device, converts numpy arrays to torch tensors.
    
    将观察数据移至指定设备，并将NumPy数组转换为PyTorch张量

    Args:
        observations: 观察数据字典
        device: 目标设备
        unsqueeze: 是否扩展张量维度
        
    Returns:
        转换后的观察数据字典，所有值都是位于指定设备的PyTorch张量
    """
    # 将每个字典值中的NumPy数组转换为PyTorch张量
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            # 根据数据类型选择适当的张量类型
            tensor_dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=tensor_dtype)
            # 如果需要，扩展张量维度（常用于添加批次维度）
            if unsqueeze:
                observations[k] = observations[k].unsqueeze(0)

    return observations


if __name__ == "__main__":
    """
    主程序入口点
    测试加载预训练的PointNavResNet策略并执行前向传播
    """
    import argparse

    parser = argparse.ArgumentParser("Load a checkpoint file for PointNavResNetPolicy")
    parser.add_argument("ckpt_path", help="path to checkpoint file")
    args = parser.parse_args()

    # 加载策略
    policy = load_pointnav_policy(args.ckpt_path)
    print("Loaded model from checkpoint successfully!")
    
    # 创建测试输入
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)
    observations = {
        "depth": torch.zeros(1, 224, 224, 1, device=torch.device("cuda")),
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),
    }
    
    # 将策略移至GPU并执行前向传播
    policy.to(torch.device("cuda"))
    action = policy.act(
        observations,
        torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32),
        torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.long),
        mask,
    )
    print("Forward pass successful!")
