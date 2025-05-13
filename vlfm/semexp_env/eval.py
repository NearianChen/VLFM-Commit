# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# 导入所需库
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from arguments import get_args  # 解析命令行参数
from envs import make_vec_envs  # 创建矢量化环境
from moviepy.editor import ImageSequenceClip  # 用于生成视频

# 导入语义探索策略模型
from vlfm.semexp_env.semexp_policy import SemExpITMPolicyV2, SemExpITMPolicyV3
# 导入工具函数
from vlfm.utils.img_utils import reorient_rescale_map, resize_images
from vlfm.utils.log_saver import is_evaluated, log_episode
from vlfm.utils.visualization import add_text_to_image

# 限制OpenMP线程数，避免多线程性能问题
os.environ["OMP_NUM_THREADS"] = "1"

# 获取参数并设置基本配置
args = get_args()
args.agent = "vlfm"  # 设置代理类型为vlfm
args.split = "val"  # 使用验证集进行评估
args.task_config = "objnav_gibson_vlfm.yaml"  # 任务配置文件
# 使用当前时间戳生成随机种子，确保每次运行结果不同
args.seed = int(time.time() * 1000) % 2**32

# 设置随机种子以确保结果的可重现性
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main() -> None:
    """主函数：执行评估过程"""
    num_episodes = int(args.num_eval_episodes)  # 要评估的场景数量
    args.device = torch.device("cuda:0" if args.cuda else "cpu")  # 设置计算设备

    # 策略模型的参数配置
    policy_kwargs = dict(
        text_prompt="Seems like there is a target_object ahead.",  # 语言提示
        pointnav_policy_path="data/pointnav_weights.pth",  # 点导航策略权重路径
        depth_image_shape=(224, 224),  # 深度图像尺寸
        pointnav_stop_radius=0.9,  # 点导航停止半径
        use_max_confidence=False,  # 是否使用最大置信度
        object_map_erosion_size=5,  # 目标图侵蚀大小参数
        exploration_thresh=0.0,  # 探索阈值
        obstacle_map_area_threshold=1.5,  # 障碍物图面积阈值(平方米)
        min_obstacle_height=0.61,  # 最小障碍物高度
        max_obstacle_height=0.88,  # 最大障碍物高度
        hole_area_thresh=100000,  # 洞区域阈值
        use_vqa=False,  # 是否使用视觉问答
        vqa_prompt="Is this ",  # 视觉问答提示
        coco_threshold=0.8,  # COCO数据集目标检测阈值
        non_coco_threshold=0.4,  # 非COCO数据集目标检测阈值
        camera_height=0.88,  # 相机高度
        min_depth=0.5,  # 最小深度
        max_depth=5.0,  # 最大深度
        camera_fov=79,  # 相机视场角
        image_width=640,  # 图像宽度
        visualize=True,  # 是否开启可视化
    )

    # 根据环境变量选择不同的策略模型
    exp_thresh = float(os.environ.get("EXPLORATION_THRESH", 0.0))
    if exp_thresh > 0.0:
        policy_cls = SemExpITMPolicyV3  # 使用V3版本的策略
        policy_kwargs["exploration_thresh"] = exp_thresh
        # 使用多个语言提示，用|分隔
        policy_kwargs["text_prompt"] = (
            "Seems like there is a target_object ahead.|There is a lot of area to explore ahead."
        )
    else:
        policy_cls = SemExpITMPolicyV2  # 使用V2版本的策略

    # 初始化策略模型
    policy = policy_cls(**policy_kwargs)  # type: ignore

    # 限制PyTorch线程数以提高性能
    torch.set_num_threads(1)
    # 创建环境
    envs = make_vec_envs(args)
    # 重置环境并获取初始观察和信息
    obs, infos = envs.reset()
    ep_id, scene_id, target_object = "", "", ""
    
    # 评估每个场景
    for ep_num in range(num_episodes):
        vis_imgs = []  # 用于存储可视化图像
        for step in range(args.max_episode_length):
            if step == 0:  # 如果是场景的第一步
                masks = torch.zeros(1, 1, device=obs.device)  # 初始掩码为0
                # 获取场景ID和目标物体
                ep_id, scene_id = infos[0]["episode_id"], infos[0]["scene_id"]
                target_object = infos[0]["goal_name"]
                print("Episode:", ep_id, "Scene:", scene_id)
            else:
                masks = torch.ones(1, 1, device=obs.device)  # 后续步骤掩码为1

            # 检查该场景是否已经被评估过
            if "ZSOS_LOG_DIR" in os.environ and is_evaluated(ep_id, scene_id):
                print(f"Episode {ep_id} in scene {scene_id} already evaluated")
                # 执行停止动作，进入下一个场景
                obs, rew, done, infos = envs.step(torch.tensor([0], dtype=torch.long))
            else:
                # 合并观察和信息
                obs_dict = merge_obs_infos(obs, infos)
                # 根据当前状态决定下一步动作
                action, policy_infos = policy.act(obs_dict, None, None, masks)

                # 如果需要生成视频，则创建并保存当前帧
                if "VIDEO_DIR" in os.environ:
                    frame = create_frame(policy_infos)
                    frame = add_text_to_image(frame, "Step: " + str(step), top=True)
                    vis_imgs.append(frame)

                action = action.squeeze(0)  # 移除批次维度
                # 执行动作
                obs, rew, done, infos = envs.step(action)

            # 如果场景结束
            if done:
                print("Success:", infos[0]["success"])  # 打印是否成功
                print("SPL:", infos[0]["spl"])  # 打印SPL指标
                # 收集结果数据
                data = {
                    "success": infos[0]["success"],  # 是否成功到达目标
                    "spl": infos[0]["spl"],  # 成功路径长度比
                    "distance_to_goal": infos[0]["distance_to_goal"],  # 到目标的距离
                    "target_object": target_object,  # 目标物体
                }
                # 如果需要，生成视频
                if "VIDEO_DIR" in os.environ:
                    try:
                        generate_video(vis_imgs, ep_id, scene_id, data)
                    except Exception:
                        print("Error generating video")
                # 如果需要且尚未评估，记录场景结果
                if "ZSOS_LOG_DIR" in os.environ and not is_evaluated(ep_id, scene_id):
                    log_episode(ep_id, scene_id, data)
                break  # 结束当前场景

    print("Test successfully completed")


def merge_obs_infos(obs: torch.Tensor, infos: Tuple[Dict, ...]) -> Dict[str, torch.Tensor]:
    """合并观察和信息到单个字典。
    
    Args:
        obs: 原始观察张量
        infos: 环境提供的额外信息
        
    Returns:
        包含处理后观察数据的字典
    """
    rgb = obs[:, :3, ...].permute(0, 2, 3, 1)  # 提取RGB信息并调整维度
    depth = obs[:, 3:4, ...].permute(0, 2, 3, 1)  # 提取深度信息并调整维度
    info_dict = infos[0]  # 获取第一个环境的信息

    def tensor_from_numpy(tensor: torch.Tensor, numpy_array: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为与指定tensor相同设备上的tensor"""
        device = tensor.device
        new_tensor = torch.from_numpy(numpy_array).to(device)
        return new_tensor

    # 构建观察字典
    obs_dict = {
        "rgb": rgb,  # RGB图像
        "depth": depth,  # 深度图像
        "objectgoal": info_dict["goal_name"].replace("-", " "),  # 目标物体名称
        "gps": tensor_from_numpy(obs, info_dict["gps"]).unsqueeze(0),  # GPS信息
        "compass": tensor_from_numpy(obs, info_dict["compass"]).unsqueeze(0),  # 罗盘信息
        "heading": tensor_from_numpy(obs, info_dict["heading"]).unsqueeze(0),  # 朝向信息
    }

    return obs_dict


def create_frame(policy_infos: Dict[str, Any]) -> np.ndarray:
    """创建可视化帧，合并多种视觉信息。
    
    Args:
        policy_infos: 策略模型产生的可视化信息
        
    Returns:
        合并后的可视化图像
    """
    vis_imgs = []
    # 处理各种可视化元素
    for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
        img = policy_infos[k]
        if "map" in k:
            img = reorient_rescale_map(img)  # 重新调整地图方向和大小
        if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
            # 如果深度图全白，添加文本说明目标未检测到
            text = "Target not currently detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            cv2.putText(
                img,
                text,
                (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1,
            )
        vis_imgs.append(img)
    # 水平堆叠所有图像，保持相同高度
    vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
    return vis_img


def generate_video(frames: List[np.ndarray], ep_id: str, scene_id: str, infos: Dict[str, Any]) -> None:
    """将图像帧序列保存为视频文件。
    
    Args:
        frames: 图像帧列表
        ep_id: 场景ID
        scene_id: 场景场景ID
        infos: 包含评估结果数据的字典
    """
    # 获取视频保存目录
    video_dir = os.environ.get("VIDEO_DIR", "video_dir")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # 提取必要信息
    episode_id = int(ep_id)
    success = int(infos["success"])
    spl = infos["spl"]
    dtg = infos["distance_to_goal"]
    goal_name = infos["target_object"]
    
    # 构建包含评估指标的文件名
    filename = (
        f"epid={episode_id:03d}-scid={scene_id}-succ={success}-spl={spl:.2f}-dtg={dtg:.2f}-target={goal_name}.mp4"
    )

    filename = os.path.join(video_dir, filename)
    # 创建10FPS的视频剪辑
    clip = ImageSequenceClip(frames, fps=10)

    # 写入视频文件
    clip.write_videofile(filename)


if __name__ == "__main__":
    main()  # 执行主函数
