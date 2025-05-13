# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

# 以下导入需要安装habitat环境，虽然本脚本未直接使用这些类，
# 但它们会注册多个类，使其可被Hydra发现。
# 此run.py脚本预期仅在已安装habitat的环境中使用，因此这些导入
# 放在这里而不是__init__.py中，避免在没有habitat的环境中（如实际部署时）
# 出现导入错误。使用noqa标记来抑制未使用导入和导入未排序的警告。
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

# 导入各种自定义组件，用于测量、观察转换、策略实现等
import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    """Habitat配置插件类
    
    为Hydra配置系统添加Habitat配置路径，确保能正确加载Habitat相关配置文件
    """
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # 将Habitat配置路径添加到搜索路径中
        search_path.append(provider="habitat", path="config/")


# 注册自定义Hydra插件，使配置系统能够找到Habitat配置文件
register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",  # 配置文件的根目录
    config_name="experiments/vlfm_objectnav_hm3d",  # 默认配置文件名
)
def main(cfg: DictConfig) -> None:
    """主函数：加载配置并执行实验
    
    Args:
        cfg: 通过Hydra加载的配置对象
    """
    # 检查数据目录是否存在
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    # 检查模型权重文件是否存在
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    # 应用配置补丁，处理配置中的默认值和依赖关系
    cfg = patch_config(cfg)
    # 使用读写模式修改配置
    with read_write(cfg):
        try:
            # 尝试移除语义传感器，可能不是所有配置都包含此传感器
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    # 执行实验：根据配置选择训练或评估模式
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    # 执行主函数
    main()