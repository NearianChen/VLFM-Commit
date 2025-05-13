# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# 此脚本用于生成一个虚拟策略模型文件，主要用于测试和初始化目的

import torch

## ---------------------debug 测试---------------------

# import os
# sub_dir = os.path.dirname(__file__)
# # 将当前脚本往前移动到上两级目录
# os.chdir(os.path.abspath(os.path.join(sub_dir, os.pardir, os.pardir)))
# # 打印当前工作目录
# print("Current working directory:", os.getcwd())
# # 将当前目录添加到Python路径中
# import sys
# sys.path.append(os.getcwd())

from vlfm.run import get_config


def save_dummy_policy(filename: str) -> None:
    """
    生成并保存一个虚拟的策略模型文件
    
    参数:
        filename: str - 要保存的文件路径
    """
    # 从配置文件加载VLFM配置
    config = get_config("config/experiments/vlfm_objectnav_hm3d.yaml")
    
    # 创建一个虚拟的模型状态字典
    dummy_dict = {
        "config": config,  # 包含模型配置信息
        "extra_state": {"step": 0},  # 额外状态信息，这里将训练步数设为0
        "state_dict": {},  # 空的模型参数字典，实际应用中这里会包含模型的权重
    }

    # 将虚拟字典保存为PyTorch模型文件
    torch.save(dummy_dict, filename)


if __name__ == "__main__":
    # 当脚本直接运行时执行以下代码
    save_dummy_policy("data/dummy_policy.pth")  # 调用函数保存虚拟策略到指定路径
    print("Dummy policy weights saved to data/dummy_policy.pth")  # 打印保存成功的提示信息
