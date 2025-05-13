# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size

from .resnet import resnet18
from .rnn_state_encoder import LSTMStateEncoder


class ResNetEncoder(nn.Module):
    """
    使用ResNet架构的视觉编码器
    从深度图像提取特征表示
    """
    visual_keys = ["depth"]  # 视觉输入类型，这里只使用深度图

    def __init__(self) -> None:
        """
        初始化ResNet编码器
        设置网络架构，包括ResNet18骨干和压缩层
        """
        super().__init__()
        self.running_mean_and_var = nn.Sequential()  # 空序列，可用于归一化处理
        self.backbone = resnet18(1, 32, 16)  # ResNet18骨干网络，输入通道为1，32个基本特征，16倍下采样
        # 压缩层，将特征图从256通道压缩到128通道
        self.compression = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 卷积层
            nn.GroupNorm(1, 128, eps=1e-05, affine=True),  # 组归一化，提高训练稳定性
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        """
        前向传播方法
        处理观察数据生成视觉特征
        
        Args:
            observations: 包含视觉输入的字典
            
        Returns:
            处理后的视觉特征
        """
        cnn_input = []  # 存储CNN输入数据
        for k in self.visual_keys:  # 遍历所有视觉输入键
            obs_k = observations[k]
            # 调整张量维度为[批次 x 通道 x 高度 x 宽度]
            obs_k = obs_k.permute(0, 3, 1, 2)
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)  # 按通道维度拼接输入
        x = F.avg_pool2d(x, 2)  # 平均池化，减少空间维度

        x = self.running_mean_and_var(x)  # 可能的归一化处理
        x = self.backbone(x)  # 通过ResNet18主干网络
        x = self.compression(x)  # 通过压缩层
        return x


class PointNavResNetNet(nn.Module):
    """
    点导航ResNet网络
    结合视觉特征、目标信息和前一动作，生成导航策略
    """
    def __init__(self, discrete_actions: bool = False, no_fwd_dict: bool = False):
        """
        初始化点导航ResNet网络
        
        Args:
            discrete_actions: 是否使用离散动作空间
            no_fwd_dict: 前向传播是否返回字典
        """
        super().__init__()
        # 根据动作类型选择不同的动作嵌入方式
        if discrete_actions:
            # 离散动作使用嵌入层，4+1表示4种动作加上一个起始标记
            self.prev_action_embedding_discrete = nn.Embedding(4 + 1, 32)
        else:
            # 连续动作使用线性层
            self.prev_action_embedding_cont = nn.Linear(in_features=2, out_features=32, bias=True)
        # 目标嵌入层，将3D目标转换为潜在表示
        self.tgt_embeding = nn.Linear(in_features=3, out_features=32, bias=True)
        # 视觉编码器和特征处理
        self.visual_encoder = ResNetEncoder()
        # 视觉特征后处理，展平并降维
        self.visual_fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),  # 展平特征图
            nn.Linear(in_features=2048, out_features=512, bias=True),  # 线性映射降维
            nn.ReLU(inplace=True),  # ReLU激活
        )
        # 状态编码器，使用LSTM处理时序信息
        self.state_encoder = LSTMStateEncoder(576, 512, 2)  # 输入576维，隐藏状态512维，2层LSTM
        self.num_recurrent_layers = self.state_encoder.num_recurrent_layers
        self.discrete_actions = discrete_actions
        self.no_fwd_dict = no_fwd_dict

    def forward(
        self,
        observations: Dict[str, torch.Tensor],  # 观察数据
        rnn_hidden_states: torch.Tensor,  # RNN隐藏状态
        prev_actions: torch.Tensor,  # 前一步动作
        masks: torch.Tensor,  # 重置掩码
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,  # RNN序列构建信息
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播方法
        处理输入数据，生成策略特征
        
        Args:
            observations: 观察数据字典
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步执行的动作
            masks: 重置掩码，0表示重置，1表示继续
            rnn_build_seq_info: 可选的RNN序列构建信息
            
        Returns:
            策略特征、更新后的RNN隐藏状态和额外信息字典
        """
        x = []  # 存储所有特征
        # 处理视觉特征
        visual_feats = self.visual_encoder(observations)
        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        # 处理目标信息，转换为更有用的表示
        goal_observations = observations["pointgoal_with_gps_compass"]
        # 转换为[距离, cos(角度), sin(角度)]形式
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],  # 距离
                torch.cos(-goal_observations[:, 1]),  # 角度的余弦
                torch.sin(-goal_observations[:, 1]),  # 角度的正弦
            ],
            -1,
        )

        x.append(self.tgt_embeding(goal_observations))  # 嵌入目标信息

        # 处理前一步动作
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # 掩码表示前一动作为零，一个额外的虚拟动作
            prev_actions = self.prev_action_embedding_discrete(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            # 连续动作空间，使用线性嵌入
            prev_actions = self.prev_action_embedding_cont(masks * prev_actions.float())

        x.append(prev_actions)  # 添加动作嵌入

        # 合并所有特征
        out = torch.cat(x, dim=1)
        # 通过状态编码器（LSTM）处理
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks, rnn_build_seq_info)

        # 根据设置决定返回格式
        if self.no_fwd_dict:
            return out, rnn_hidden_states  # type: ignore

        return out, rnn_hidden_states, {}


class CustomNormal(torch.distributions.normal.Normal):
    """
    自定义正态分布类
    重写sample方法使用可重参数化技巧
    这对于策略梯度方法中保持梯度流动很重要
    """
    def sample(self, sample_shape: Size = torch.Size()) -> torch.Tensor:
        """
        使用可重参数化采样
        
        Args:
            sample_shape: 采样形状
            
        Returns:
            采样的动作
        """
        return self.rsample(sample_shape)  # 使用rsample而非sample，保持梯度流动


class GaussianNet(nn.Module):
    """
    高斯网络类
    生成动作的均值和标准差，构建动作分布
    """
    min_log_std: int = -5  # 标准差对数的最小值
    max_log_std: int = 2  # 标准差对数的最大值
    log_std_init: float = 0.0  # 标准差对数的初始值

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """
        初始化高斯网络
        
        Args:
            num_inputs: 输入特征维度
            num_outputs: 输出动作维度
        """
        super().__init__()
        num_linear_outputs = 2 * num_outputs  # 输出包括均值和标准差

        # 线性层，同时预测均值和标准差
        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)  # 正交初始化权重
        nn.init.constant_(self.mu_maybe_std.bias, 0)  # 均值偏置初始化为0
        nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], self.log_std_init)  # 标准差偏置初始化为预设值

    def forward(self, x: torch.Tensor) -> CustomNormal:
        """
        前向传播方法
        从输入特征生成动作分布
        
        Args:
            x: 输入特征
            
        Returns:
            动作的正态分布
        """
        mu_maybe_std = self.mu_maybe_std(x).float()  # 预测均值和标准差
        mu, std = torch.chunk(mu_maybe_std, 2, -1)  # 分割为均值和标准差

        mu = torch.tanh(mu)  # 使用tanh限制均值在[-1,1]范围

        # 限制标准差的范围并取指数
        std = torch.clamp(std, self.min_log_std, self.max_log_std)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)  # 返回自定义正态分布


class PointNavResNetPolicy(nn.Module):
    """
    点导航ResNet策略
    将特征提取网络与动作分布网络结合，实现端到端导航策略
    """
    def __init__(self) -> None:
        """
        初始化点导航ResNet策略
        创建网络和动作分布模块
        """
        super().__init__()
        self.net = PointNavResNetNet()  # 特征提取网络
        self.action_distribution = GaussianNet(512, 2)  # 动作分布网络，输入512维，输出2维动作(线速度和角速度)

    def act(
        self,
        observations: Dict[str, torch.Tensor],  # 观察数据
        rnn_hidden_states: torch.Tensor,  # RNN隐藏状态
        prev_actions: torch.Tensor,  # 前一步动作
        masks: torch.Tensor,  # 重置掩码
        deterministic: bool = False,  # 是否使用确定性策略
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行策略动作
        根据当前观察和状态，生成下一步动作
        
        Args:
            observations: 观察数据字典，包含深度图像和目标信息
            rnn_hidden_states: RNN隐藏状态
            prev_actions: 前一步执行的动作
            masks: 重置掩码，0表示重置，1表示继续
            deterministic: 是否使用确定性策略(均值)而非随机采样
            
        Returns:
            动作张量和更新后的RNN隐藏状态
        """
        # 通过网络获取特征
        features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        # 生成动作分布
        distribution = self.action_distribution(features)

        # 确定性模式使用分布均值，否则进行采样
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        return action, rnn_hidden_states


if __name__ == "__main__":
    """
    主程序入口
    用于测试加载预训练模型并执行前向传播
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_path", type=str, help="Path to state_dict file")
    args = parser.parse_args()

    # 加载模型权重
    ckpt = torch.load(args.state_dict_path, map_location="cpu")
    policy = PointNavResNetPolicy()
    print(policy)
    current_state_dict = policy.state_dict()
    policy.load_state_dict({k: v for k, v in ckpt.items() if k in current_state_dict})
    print("Loaded model from checkpoint successfully!")

    # 将模型移动到GPU
    policy = policy.to(torch.device("cuda"))
    print("Successfully moved model to GPU!")

    # 创建测试输入
    observations = {
        "depth": torch.ones(1, 212, 240, 1, device=torch.device("cuda")),  # 批次大小1，高212，宽240的深度图
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),  # 目标位置信息
    }
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)  # 重置掩码

    # 创建初始RNN状态
    rnn_state = torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32)

    # 执行前向传播
    action = policy.act(
        observations,
        rnn_state,
        torch.zeros(1, 2, device=torch.device("cuda"), dtype=torch.float32),  # 初始动作为零
        mask,
        deterministic=True,  # 使用确定性策略
    )

    print("Forward pass successful!")
    print(action[0].detach().cpu().numpy())  # 打印生成的动作
