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


# 导入基础配置类
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class XBotLCfg(LeggedRobotCfg):
    """
    XBotL人形机器人的配置类
    包含机器人的所有仿真和训练参数设置
    """
    class env(LeggedRobotCfg.env):
        """环境基础配置"""
        # 观测维度相关配置
        frame_stack = 15              # 历史帧堆叠数量，用于时序信息
        c_frame_stack = 3             # critic网络历史帧堆叠数量
        num_single_obs = 47           # 单帧观测向量的维度
        num_observations = int(frame_stack * num_single_obs)  # 总观测维度
        single_num_privileged_obs = 73  # 单帧特权观测维度（真实状态信息）
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)  # 总特权观测维度
        num_actions = 12              # 动作空间维度（12个关节）
        num_envs = 4096              # 并行环境数量
        episode_length_s = 24        # 每个回合的长度（秒）
        use_ref_actions = False      # 是否使用参考动作加速训练

    class safety:
        """安全限制配置"""
        # 安全系数设置
        pos_limit = 1.0      # 位置限制系数
        vel_limit = 1.0      # 速度限制系数
        torque_limit = 0.85  # 扭矩限制系数

    class asset(LeggedRobotCfg.asset):
        """机器人资产配置"""
        # URDF文件路径
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/urdf/XBot-L.urdf'

        name = "XBot-L"              # 机器人名称
        foot_name = "ankle_roll"     # 脚部关节名称
        knee_name = "knee"           # 膝关节名称

        # 碰撞和终止条件
        terminate_after_contacts_on = ['base_link']  # 接触这些部位时终止回合
        penalize_contacts_on = ["base_link"]        # 接触这些部位时给予惩罚
        self_collisions = 0                         # 自碰撞检测（1=禁用，0=启用）
        flip_visual_attachments = False             # 是否翻转视觉附件
        replace_cylinder_with_capsule = False       # 是否用胶囊体替换圆柱体
        fix_base_link = False                       # 是否固定基座连杆

    class terrain(LeggedRobotCfg.terrain):
        """地形配置"""
        mesh_type = 'plane'          # 地形网格类型：'plane'(平面)，'trimesh'(三角网格)
        # mesh_type = 'trimesh'      # 可选：三角网格地形
        curriculum = False           # 是否启用课程学习
        
        # 粗糙地形专用设置
        measure_heights = False      # 是否测量高度信息
        static_friction = 0.6        # 静摩擦系数
        dynamic_friction = 0.6       # 动摩擦系数
        terrain_length = 8.          # 地形长度（米）
        terrain_width = 8.           # 地形宽度（米）
        # 一个网格里有多少个机器人
        num_rows = 20               # 地形行数
        num_cols = 20               # 地形列数
        max_init_terrain_level = 10  # 起始课程难度级别
        
        # 地形类型比例：平面；障碍；不平整；上坡；下坡；上楼梯；下楼梯
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.            # 恢复系数（弹性）

    class noise:
        """噪声配置"""
        add_noise = True         # 是否添加噪声
        noise_level = 0.6        # 噪声级别（缩放其他数值）

        class noise_scales:
            """各种观测量的噪声缩放系数"""
            dof_pos = 0.05              # 关节位置噪声
            dof_vel = 0.5               # 关节速度噪声
            ang_vel = 0.1               # 角速度噪声
            lin_vel = 0.05              # 线速度噪声
            quat = 0.03                 # 四元数（姿态）噪声
            height_measurements = 0.1    # 高度测量噪声

    class init_state(LeggedRobotCfg.init_state):
        """初始状态配置"""
        pos = [0.0, 0.0, 0.95]  # 初始位置 [x, y, z]（米）

        # 默认关节角度（弧度）- 当动作为0.0时的目标角度
        default_joint_angles = {
            # 左腿关节
            'left_leg_roll_joint': 0.,      # 左腿滚转关节
            'left_leg_yaw_joint': 0.,       # 左腿偏航关节
            'left_leg_pitch_joint': 0.,     # 左腿俯仰关节
            'left_knee_joint': 0.,          # 左膝关节
            'left_ankle_pitch_joint': 0.,   # 左踝俯仰关节
            'left_ankle_roll_joint': 0.,    # 左踝滚转关节
            
            # 右腿关节
            'right_leg_roll_joint': 0.,     # 右腿滚转关节
            'right_leg_yaw_joint': 0.,      # 右腿偏航关节
            'right_leg_pitch_joint': 0.,    # 右腿俯仰关节
            'right_knee_joint': 0.,         # 右膝关节
            'right_ankle_pitch_joint': 0.,  # 右踝俯仰关节
            'right_ankle_roll_joint': 0.,   # 右踝滚转关节
        }

    class control(LeggedRobotCfg.control):
        """控制系统配置"""
        # PD控制器参数
        stiffness = {
            'leg_roll': 200.0,    # 腿部滚转关节刚度
            'leg_pitch': 350.0,   # 腿部俯仰关节刚度
            'leg_yaw': 200.0,     # 腿部偏航关节刚度
            'knee': 350.0,        # 膝关节刚度
            'ankle': 15           # 踝关节刚度
        }
        damping = {
            'leg_roll': 10,       # 腿部滚转关节阻尼
            'leg_pitch': 10,      # 腿部俯仰关节阻尼
            'leg_yaw': 10,        # 腿部偏航关节阻尼
            'knee': 10,           # 膝关节阻尼
            'ankle': 10           # 踝关节阻尼
        }

        # 动作缩放：目标角度 = 动作缩放 * 动作 + 默认角度
        action_scale = 0.25
        # 抽取：每个策略时间步内的控制动作更新次数
        decimation = 10  # 100Hz控制频率

    class sim(LeggedRobotCfg.sim):
        """仿真配置"""
        dt = 0.001       # 仿真时间步长：1000 Hz
        substeps = 1     # 子步数
        up_axis = 1      # 向上轴：0=y轴，1=z轴

        class physx(LeggedRobotCfg.sim.physx):
            """PhysX物理引擎参数"""
            num_threads = 10                      # 线程数
            solver_type = 1                       # 求解器类型：0=PGS，1=TGS
            num_position_iterations = 4           # 位置迭代次数
            num_velocity_iterations = 1           # 速度迭代次数
            contact_offset = 0.01                 # 接触偏移量（米）
            rest_offset = 0.0                     # 静止偏移量（米）
            bounce_threshold_velocity = 0.1       # 弹跳阈值速度（米/秒）
            max_depenetration_velocity = 1.0      # 最大去穿透速度
            max_gpu_contact_pairs = 2**23         # 最大GPU接触对数（8000+环境需要2**24）
            default_buffer_size_multiplier = 5    # 默认缓冲区大小倍数
            # 接触收集模式：0=从不，1=最后子步，2=所有子步（默认=2）
            contact_collection = 2

    class domain_rand:
        """域随机化配置 - 增强训练鲁棒性"""
        # 摩擦系数随机化
        randomize_friction = True           # 是否随机化摩擦系数
        friction_range = [0.1, 2.0]        # 摩擦系数范围
        
        # 质量随机化
        randomize_base_mass = True          # 是否随机化基座质量
        added_mass_range = [-5., 5.]       # 附加质量范围（千克）
        
        # 推力扰动
        push_robots = True                  # 是否对机器人施加推力
        push_interval_s = 4                 # 推力间隔时间（秒）
        max_push_vel_xy = 0.2              # 最大推力线速度（米/秒）
        max_push_ang_vel = 0.4             # 最大推力角速度（弧度/秒）
        
        # 动态随机化
        action_delay = 0.5                  # 动作延迟系数
        action_noise = 0.02                 # 动作噪声强度

    class commands(LeggedRobotCfg.commands):
        """命令配置 - 控制机器人的运动指令"""
        # 命令向量：lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        # （在航向模式下，ang_vel_yaw会根据航向误差重新计算）
        num_commands = 4             # 命令数量
        resampling_time = 8.         # 命令更新间隔时间（秒）
        heading_command = True       # 是否启用航向命令模式

        class ranges:
            """各命令的取值范围"""
            lin_vel_x = [-0.3, 0.6]      # x方向线速度范围（米/秒）
            lin_vel_y = [-0.3, 0.3]      # y方向线速度范围（米/秒）
            ang_vel_yaw = [-0.3, 0.3]    # 偏航角速度范围（弧度/秒）
            heading = [-3.14, 3.14]      # 航向角范围（弧度）

    class rewards:
        """奖励函数配置"""
        base_height_target = 0.89        # 目标基座高度（米）
        min_dist = 0.2                   # 双脚最小距离（米）
        max_dist = 0.5                   # 双脚最大距离（米）
        
        # LLM参数调优相关设置
        target_joint_pos_scale = 0.17   # 目标关节位置缩放（弧度）
        target_feet_height = 0.06       # 目标脚部抬起高度（米）
        cycle_time = 0.64               # 步态周期时间（秒）
        
        # 奖励计算策略
        only_positive_rewards = True     # 是否将负奖励截断为零（避免早期终止问题）
        tracking_sigma = 5               # 跟踪奖励计算：exp(error*sigma)
        max_contact_force = 700          # 最大接触力，超过此值将被惩罚

        class scales:
            """各奖励项的权重缩放系数"""
            # 参考运动跟踪
            joint_pos = 1.6                # 关节位置跟踪奖励
            feet_clearance = 1.             # 脚部离地间隙奖励
            feet_contact_number = 1.2       # 脚部接触数量奖励
            
            # 步态相关
            feet_air_time = 1.              # 脚部腾空时间奖励
            foot_slip = -0.05               # 脚部滑动惩罚
            feet_distance = 0.2             # 双脚距离奖励
            knee_distance = 0.2             # 双膝距离奖励
            
            # 接触力相关
            feet_contact_forces = -0.01     # 脚部接触力惩罚
            
            # 速度跟踪
            tracking_lin_vel = 1.2          # 线速度跟踪奖励
            tracking_ang_vel = 1.1          # 角速度跟踪奖励
            vel_mismatch_exp = 0.5          # 速度不匹配惩罚（z方向线速度；x,y方向角速度）
            low_speed = 0.2                 # 低速奖励
            track_vel_hard = 0.5            # 硬约束速度跟踪奖励
            
            # 基座位置和姿态
            default_joint_pos = 0.5         # 默认关节位置奖励
            orientation = 1.                # 姿态保持奖励
            base_height = 0.2               # 基座高度奖励
            base_acc = 0.2                  # 基座加速度惩罚
            
            # 能耗相关
            action_smoothness = -0.002      # 动作平滑性惩罚
            torques = -1e-5                 # 扭矩惩罚
            dof_vel = -5e-4                 # 关节速度惩罚
            dof_acc = -1e-7                 # 关节加速度惩罚
            collision = -1.                 # 碰撞惩罚

    class normalization:
        """归一化配置"""
        class obs_scales:
            """观测量缩放系数"""
            lin_vel = 2.                # 线速度缩放
            ang_vel = 1.                # 角速度缩放
            dof_pos = 1.                # 关节位置缩放
            dof_vel = 0.05              # 关节速度缩放
            quat = 1.                   # 四元数缩放
            height_measurements = 5.0    # 高度测量缩放
        
        clip_observations = 18.     # 观测值裁剪范围
        clip_actions = 18.          # 动作值裁剪范围


class XBotLCfgPPO(LeggedRobotCfgPPO):
    """
    XBotL机器人的PPO算法配置类
    包含强化学习训练的所有超参数设置
    """
    seed = 5                              # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 训练器类名

    class policy:
        """策略网络配置"""
        init_noise_std = 1.0                  # 初始策略噪声标准差
        actor_hidden_dims = [512, 256, 128]   # Actor网络隐藏层维度
        critic_hidden_dims = [768, 256, 128]  # Critic网络隐藏层维度

    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO算法超参数"""
        entropy_coef = 0.001        # 熵系数（探索性）
        learning_rate = 1e-5        # 学习率
        num_learning_epochs = 2     # 每次更新的学习轮数
        gamma = 0.994              # 折扣因子
        lam = 0.9                  # GAE lambda参数
        num_mini_batches = 4       # 小批次数量

    class runner:
        """训练运行器配置"""
        policy_class_name = 'ActorCritic'     # 策略类名
        algorithm_class_name = 'PPO'          # 算法类名
        num_steps_per_env = 60                # 每个环境每次迭代的步数
        max_iterations = 3001                 # 最大策略更新次数

        # 日志记录配置
        save_interval = 100                   # 模型保存间隔（每100次迭代检查保存）
        experiment_name = 'XBot_ppo'          # 实验名称
        run_name = ''                         # 运行名称
        
        # 加载和恢复配置
        resume = False                        # 是否恢复训练
        load_run = -1                         # 要加载的运行（-1=最后一次运行）
        checkpoint = -1                       # 要加载的检查点（-1=最后保存的模型）
        resume_path = None                    # 恢复路径（由load_run和checkpoint更新）
