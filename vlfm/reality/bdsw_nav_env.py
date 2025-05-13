# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from spot_wrapper.spot import Spot

from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.reality.pointnav_env import PointNavEnv
from vlfm.reality.robots.bdsw_robot import BDSWRobot


def run_env(env: PointNavEnv, policy: WrappedPointNavResNetPolicy, goal: np.ndarray) -> None:
    """
    运行导航环境，使用指定的策略控制机器人到达目标位置

    Args:
        env: 点导航环境实例
        policy: 包装后的PointNav ResNet策略
        goal: 目标位置坐标 [x, y]
    """
    observations = env.reset(goal)  # 重置环境并设置目标位置，获取初始观察
    done = False  # 任务完成标志
    mask = torch.zeros(1, 1, device=policy.device, dtype=torch.bool)  # 创建初始掩码，首次迭代置为0
    action = policy.act(observations, mask)  # 根据观察获取第一个动作
    action_dict = {"rho_theta": action}  # 构建动作字典，包含距离和角度信息
    while not done:
        observations, _, done, info = env.step(action_dict)  # 执行动作，获取新的观察和状态
        action = policy.act(observations, mask, deterministic=True)  # 确定性地选择下一个动作
        mask = torch.ones_like(mask)  # 更新掩码，后续迭代置为1


if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pointnav_ckpt_path",
        type=str,
        default="pointnav_resnet_18.pth",
        help="Path to the pointnav model checkpoint",  # 点导航模型检查点路径
    )
    parser.add_argument(
        "-g",
        "--goal",
        type=str,
        default="3.5,0.0",
        help="Goal location in the form x,y",  # 目标位置，格式为x,y
    )
    args = parser.parse_args()
    pointnav_ckpt_path = args.pointnav_ckpt_path  # 获取模型路径参数
    policy = WrappedPointNavResNetPolicy(pointnav_ckpt_path)  # 初始化包装后的点导航策略
    goal = np.array([float(x) for x in args.goal.split(",")])  # 解析目标坐标

    spot = Spot("BDSW_env")  # 创建Spot机器人实例，名称可以任意设置
    with spot.get_lease():  # 获取机器人控制权，任务完成或出错时自动释放
        spot.power_on()  # 启动机器人电源
        spot.blocking_stand()  # 使机器人站立，这是一个阻塞调用
        robot = BDSWRobot(spot)  # 创建Boston Dynamics Spot包装机器人实例
        env = PointNavEnv(robot)  # 创建点导航环境
        run_env(env, policy, goal)  # 运行导航任务
