# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

# 导入必要的库
import datetime  # 日期时间处理
import os        # 操作系统接口
import copy      # 深拷贝功能
import torch     # PyTorch深度学习框架
import numpy as np    # 数值计算库
import random    # 随机数生成
from isaacgym import gymapi    # Isaac Gym API
from isaacgym import gymutil   # Isaac Gym 工具函数

# 导入项目根目录和环境目录
from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    """将类对象转换为字典格式
    
    Args:
        obj: 要转换的类对象
        
    Returns:
        dict: 转换后的字典
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    # 遍历对象的所有属性
    for key in dir(obj):
        if key.startswith("_"):  # 跳过私有属性
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            # 递归处理列表中的每个元素
            for item in val:
                element.append(class_to_dict(item))
        else:
            # 递归处理单个元素
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    """从字典更新类对象的属性
    
    Args:
        obj: 要更新的类对象
        dict: 包含新属性值的字典
    """
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            # 如果属性是类型，递归更新
            update_class_from_dict(attr, val)
        else:
            # 直接设置属性值
            setattr(obj, key, val)
    return


def set_seed(seed):
    """设置随机种子，确保实验的可重复性
    
    Args:
        seed: 随机种子值，如果为-1则随机生成
    """
    if seed == -1:
        # 如果种子为-1，随机生成一个种子
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    # 设置各个库的随机种子
    random.seed(seed)           # Python random模块
    np.random.seed(seed)        # NumPy随机数生成器
    torch.manual_seed(seed)     # PyTorch CPU随机数生成器
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python哈希随机化
    torch.cuda.manual_seed(seed)              # PyTorch GPU随机数生成器（单GPU）
    torch.cuda.manual_seed_all(seed)          # PyTorch GPU随机数生成器（多GPU）


def parse_sim_params(args, cfg):
    """解析和配置仿真参数
    
    Args:
        args: 命令行参数
        cfg: 配置字典
        
    Returns:
        sim_params: 配置好的仿真参数对象
    """
    # Isaac Gym Preview 2的代码
    # 初始化仿真参数
    sim_params = gymapi.SimParams()

    # 根据命令行参数设置一些值
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("警告: 使用Flex引擎配合GPU而不是PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # 配置PhysX引擎参数
        sim_params.physx.use_gpu = args.use_gpu          # 是否使用GPU
        sim_params.physx.num_subscenes = args.subscenes  # 子场景数量
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline  # 是否使用GPU管道

    # 如果配置文件中提供了仿真选项，解析并更新/覆盖上述设置
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # 如果命令行中传递了线程数，覆盖默认设置
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    """获取模型加载路径
    
    Args:
        root: 根目录路径
        load_run: 要加载的运行目录，-1表示最新的
        checkpoint: 检查点编号，-1表示最新的
        
    Returns:
        load_path: 模型文件的完整路径
    """
    def month_to_number(month):
        """将月份缩写转换为数字"""
        return datetime.datetime.strptime(month, "%b").month

    try:
        # 获取所有运行目录
        runs = os.listdir(root)
        try:
            # 按月份、日期、时间排序
            runs.sort(key=lambda x: (month_to_number(x[:3]), int(x[3:5]), x[6:]))
        except ValueError as e:
            print("警告 - 无法按月份排序运行目录: " + str(e))
            runs.sort()
        # 移除exported目录（如果存在）
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])  # 最新的运行目录
    except:
        raise ValueError("该目录中没有运行记录: " + root)
    
    # 确定要加载的运行目录
    if load_run == -1:
        load_run = last_run  # 使用最新的运行
    else:
        load_run = os.path.join(root, load_run)  # 使用指定的运行
    
    # 确定要加载的模型文件
    if checkpoint == -1:
        # 查找所有模型文件并选择最新的
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))  # 按名称排序
        model = models[-1]  # 选择最新的模型
    else:
        # 使用指定的检查点
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    """根据命令行参数更新配置
    
    Args:
        env_cfg: 环境配置对象
        cfg_train: 训练配置对象
        args: 命令行参数
        
    Returns:
        tuple: 更新后的环境配置和训练配置
    """
    # 更新环境配置
    if env_cfg is not None:
        # 更新环境数量
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    
    # 更新训练配置
    if cfg_train is not None:
        # 更新随机种子
        if args.seed is not None:
            cfg_train.seed = args.seed
        
        # 更新算法运行器参数
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations  # 最大迭代次数
        if args.resume:
            cfg_train.runner.resume = args.resume                  # 是否恢复训练
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name  # 实验名称
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name              # 运行名称
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run              # 要加载的运行
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint          # 检查点编号

    return env_cfg, cfg_train


def get_args():
    """解析命令行参数
    
    Returns:
        args: 解析后的命令行参数对象
    """
    # 定义自定义参数列表
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "XBotL_free",
            "help": "任务名称，从检查点恢复训练或开始测试。如果提供则覆盖配置文件。",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "从检查点恢复训练",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "要运行或加载的实验名称。如果提供则覆盖配置文件。",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "运行名称。如果提供则覆盖配置文件。",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "当resume=True时要加载的运行名称。如果为-1：将加载最后一次运行。如果提供则覆盖配置文件。",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "保存的模型检查点编号。如果为-1：将加载最后一个检查点。如果提供则覆盖配置文件。",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "强制始终关闭显示",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "使用horovod进行多GPU训练",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "强化学习算法使用的设备（cpu, gpu, cuda:0, cuda:1 等）",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "要创建的环境数量。如果提供则覆盖配置文件。",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "随机种子。如果提供则覆盖配置文件。",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "最大训练迭代次数。如果提供则覆盖配置文件。",
        },
    ]
    # 解析参数
    args = gymutil.parse_arguments(
        description="强化学习策略", custom_parameters=custom_parameters
    )

    # 名称对齐
    args.sim_device_id = args.compute_device_id    # 仿真设备ID
    args.sim_device = args.sim_device_type         # 仿真设备类型
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"  # 添加设备ID
    return args


def export_policy_as_jit(actor_critic, path):
    """将策略模型导出为JIT脚本格式
    
    Args:
        actor_critic: actor-critic模型对象
        path: 导出路径
    """
    # 创建导出目录（如果不存在）
    os.makedirs(path, exist_ok=True)
    # 构建完整的文件路径
    path = os.path.join(path, "policy_1.pt")
    # 深拷贝actor模型并移至CPU
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    # 将模型转换为JIT脚本
    traced_script_module = torch.jit.script(model)
    # 保存JIT脚本模型
    traced_script_module.save(path)
