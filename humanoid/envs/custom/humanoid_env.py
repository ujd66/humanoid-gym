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


# 导入所需的模块和库
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain


class XBotLFreeEnv(LeggedRobot):
    '''
    XBotL自由环境类 - 用于双足机器人的自定义强化学习环境

    参数:
        cfg (LeggedRobotCfg): 双足机器人的配置对象
        sim_params: 仿真参数
        physics_engine: 使用的物理引擎
        sim_device: 仿真设备
        headless: 是否以无头模式运行仿真

    属性:
        last_feet_z (float): 上一次脚部z坐标位置
        feet_height (torch.Tensor): 表示脚部高度的张量
        sim (gymtorch.GymSim): 仿真对象
        terrain (HumanoidTerrain): 地形对象
        up_axis_idx (int): 向上轴的索引
        command_input (torch.Tensor): 命令输入张量
        privileged_obs_buf (torch.Tensor): 特权观测缓冲区张量
        obs_buf (torch.Tensor): 观测缓冲区张量
        obs_history (collections.deque): 包含观测历史的双端队列
        critic_history (collections.deque): 包含评价器观测历史的双端队列

    方法:
        _push_robots(): 通过设置随机基座速度来随机推动机器人
        _get_phase(): 计算步态周期的相位
        _get_gait_phase(): 计算步态相位
        compute_ref_state(): 计算参考状态
        create_sim(): 创建仿真、地形和环境
        _get_noise_scale_vec(cfg): 设置用于缩放添加到观测中的噪声的向量
        step(actions): 使用给定动作执行仿真步骤
        compute_observations(): 计算观测
        reset_idx(env_ids): 重置指定环境ID的环境
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """
        初始化XBotL环境
        
        参数:
            cfg: 机器人配置对象
            sim_params: 仿真参数
            physics_engine: 物理引擎
            sim_device: 仿真设备
            headless: 是否无头模式
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05  # 上一次脚部z坐标位置
        # 初始化脚部高度张量（每个环境两只脚）
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        # 重置所有环境
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        # 计算初始观测
        self.compute_observations()

    def _push_robots(self):
        """
        随机推动机器人。通过设置随机基座速度来模拟冲击力。
        这可以增强机器人对外界扰动的鲁棒性。
        """
        # 获取最大推力参数
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        
        # 设置随机线性推力（x和y方向）
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        # 设置随机角度推力
        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)
        self.root_states[:, 10:13] = self.rand_push_torque

        # 应用推力到仿真中
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        """
        获取当前步态周期的相位
        
        返回:
            phase: 步态相位（0-1之间的值）
        """
        cycle_time = self.cfg.rewards.cycle_time  # 步态周期时间
        phase = self.episode_length_buf * self.dt / cycle_time  # 计算相位
        return phase

    def _get_gait_phase(self):
        """
        获取步态相位掩码
        
        返回:
            stance_mask: 步态掩码，1表示支撑相，0表示摆动相
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)  # 正弦函数计算相位
        
        # 初始化支撑掩码
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 左脚支撑相（正弦值大于等于0）
        stance_mask[:, 0] = sin_pos >= 0
        # 右脚支撑相（正弦值小于0）
        stance_mask[:, 1] = sin_pos < 0
        # 双支撑相（在正弦值接近0时）
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        """
        计算参考状态（用于步态跟踪的参考关节位置）
        根据步态相位生成各关节的目标位置，实现类似行走的步态模式
        """
        phase = self._get_phase()  # 获取当前步态相位
        sin_pos = torch.sin(2 * torch.pi * phase)  # 正弦函数计算
        sin_pos_l = sin_pos.clone()  # 左脚相位
        sin_pos_r = sin_pos.clone()  # 右脚相位
        
        # 初始化参考关节位置
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale  # 第一级缩放系数
        scale_2 = 2 * scale_1  # 第二级缩放系数（用于膝关节）
        
        # 左脚支撑相时设置为默认关节位置（摆动相时设置抛物线）
        sin_pos_l[sin_pos_l > 0] = 0  # 支撑相时不动
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1    # 左腿俯仰关节
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2    # 左膝关节（更大的弯曲）
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1    # 左踝俯仰关节
        
        # 右脚支撑相时设置为默认关节位置（摆动相时设置抛物线）
        sin_pos_r[sin_pos_r < 0] = 0  # 支撑相时不动
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1    # 右腿俯仰关节
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2    # 右膝关节（更大的弯曲）
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1   # 右踝俯仰关节
        
        # 双支撑相时设置为0（站立姿态）
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        # 计算参考动作（放大两倍）
        self.ref_action = 2 * self.ref_dof_pos


    def create_sim(self):
        """
        创建仿真、地形和环境
        负责初始化Isaac Gym仿真环境的所有组件
        """
        # 设置向上轴索引（2代表z轴，1代表y轴）
        self.up_axis_idx = 2
        
        # 创建仿真实例
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # 获取地形类型并创建相应地形
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            # 创建复杂地形（高度场或三角网格）
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
            
        # 根据地形类型创建对应的地面
        if mesh_type == 'plane':
            self._create_ground_plane()      # 创建平面地面
        elif mesh_type == 'heightfield':
            self._create_heightfield()       # 创建高度场地形
        elif mesh_type == 'trimesh':
            self._create_trimesh()           # 创建三角网格地形
        elif mesh_type is not None:
            raise ValueError(
                "地形网格类型不可识别。允许的类型为 [None, plane, heightfield, trimesh]")
        
        # 创建环境实例
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """
        设置用于缩放添加到观测中的噪声的向量
        [注意]: 在更改观测结构时必须适应此方法

        参数:
            cfg (Dict): 环境配置文件

        返回:
            torch.Tensor: 用于乘以[-1, 1]中均匀分布的缩放向量
        """
        # 初始化噪声向量
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        
        # 为不同观测维度设置噪声缩放系数
        noise_vec[0: 5] = 0.                                                            # 命令（不加噪声）
        noise_vec[5: 17] = noise_scales.dof_pos * self.obs_scales.dof_pos              # 关节位置
        noise_vec[17: 29] = noise_scales.dof_vel * self.obs_scales.dof_vel             # 关节速度
        noise_vec[29: 41] = 0.                                                          # 上一次动作（不加噪声）
        noise_vec[41: 44] = noise_scales.ang_vel * self.obs_scales.ang_vel             # 角速度
        noise_vec[44: 47] = noise_scales.quat * self.obs_scales.quat                   # 欧拉角x,y
        return noise_vec


    def step(self, actions):
        """
        执行一个仿真步骤
        
        参数:
            actions: 动作张量，包含所有关节的目标动作
            
        返回:
            仿真步骤的结果
        """
        # 如果启用参考动作，将参考动作添加到当前动作
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
            
        # 裁剪动作在允许范围内
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        
        # 动态随机化 - 模拟动作延迟和噪声
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions  # 动作延迟
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions  # 动作噪声
        
        return super().step(actions)


    def compute_observations(self):
        """
        计算观测向量
        将机器人的各种状态信息组合成用于策略网络和评价网络的观测
        """
        # 获取当前步态相位并计算参考状态
        phase = self._get_phase()
        self.compute_ref_state()

        # 计算步态相位的正弦和余弦值（用于编码周期性信息）
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        # 获取步态相位和接触状态
        stance_mask = self._get_gait_phase()  # 步态相位掩码
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.  # 脚部接触掩码

        # 组合命令输入（包含步态相位和运动命令）
        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        # 计算归一化后的关节位置和速度
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos  # 关节位置偏移
        dq = self.dof_vel * self.obs_scales.dof_vel  # 关节速度
        
        # 计算当前关节位置与参考位置的差异
        diff = self.dof_pos - self.ref_dof_pos

        # 构建特权观测缓冲区（包含真实状态信息，用于评价网络）
        self.privileged_obs_buf = torch.cat((
            self.command_input,                                                     # 命令输入 (5D)
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 关节位置偏移 (12D)
            self.dof_vel * self.obs_scales.dof_vel,                                # 关节速度 (12D)
            self.actions,                                                          # 当前动作 (12D)
            diff,                                                                  # 与参考的差异 (12D)
            self.base_lin_vel * self.obs_scales.lin_vel,                          # 基座线速度 (3D)
            self.base_ang_vel * self.obs_scales.ang_vel,                          # 基座角速度 (3D)
            self.base_euler_xyz * self.obs_scales.quat,                           # 基座欧拉角 (3D)
            self.rand_push_force[:, :2],                                          # 随机推力 (2D)
            self.rand_push_torque,                                                # 随机扭矩 (3D)
            self.env_frictions,                                                   # 环境摩擦系数 (1D)
            self.body_mass / 30.,                                                 # 身体质量 (1D)
            stance_mask,                                                          # 步态掩码 (2D)
            contact_mask,                                                         # 接触掩码 (2D)
        ), dim=-1)

        # 构建主观测缓冲区（用于策略网络）
        obs_buf = torch.cat((
            self.command_input,                                    # 命令输入 (5D)
            q,                                                    # 关节位置偏移 (12D)
            dq,                                                   # 关节速度 (12D)
            self.actions,                                         # 当前动作 (12D)
            self.base_ang_vel * self.obs_scales.ang_vel,         # 基座角速度 (3D)
            self.base_euler_xyz * self.obs_scales.quat,          # 基座欧拉角 (3D)
        ), dim=-1)

        # 如果启用高度测量，将高度信息添加到特权观测中
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # 添加噪声（如果启用）
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
            
        # 更新观测历史
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        # 将历史观测堆叠成最终的观测向量
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K (展平为一维)
        # 为评价网络组合最近的特权观测
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """
        重置指定环境的状态
        
        参数:
            env_ids: 要重置的环境ID列表
        """
        super().reset_idx(env_ids)
        # 清零观测历史
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        # 清零评价器历史
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ 奖励函数 ================================================== #
    def _reward_joint_pos(self):
        """
        基于当前关节位置与目标关节位置的差异计算奖励
        奖励机器人跟踪参考步态，促进自然的行走模式
        
        返回:
            r: 关节位置跟踪奖励
        """
        joint_pos = self.dof_pos.clone()      # 当前关节位置
        pos_target = self.ref_dof_pos.clone() # 参考关节位置
        diff = joint_pos - pos_target          # 位置差异
        # 使用指数函数奖励小差异，同时惩罚大差异
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        基于双脚之间距离计算奖励。惩罚双脚过近或过远的情况
        保持合理的步幅，促进稳定和自然的行走姿态
        
        返回:
            双脚距离奖励
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]  # 获取双脚位置(x,y)
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)  # 计算双脚距离
        fd = self.cfg.rewards.min_dist      # 最小允许距离
        max_df = self.cfg.rewards.max_dist  # 最大允许距离
        
        # 计算距离过近和过远的惩罚
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)        # 距离过近的惩罚
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)      # 距离过远的惩罚
        # 使用指数函数给出奖励，距离在合理范围内时奖励最大
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        基于人形机器人双膝之间距离计算奖励
        防止双腿过度内外撰，保持自然的腿部姿态
        
        返回:
            双膝距离奖励
        """
        knee_pos = self.rigid_state[:, self.knee_indices, :2]  # 获取双膝位置(x,y)
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)  # 计算双膝距离
        fd = self.cfg.rewards.min_dist        # 最小允许距离
        max_df = self.cfg.rewards.max_dist / 2  # 最大允许距离（膝部比脚部要更近）
        
        # 计算距离过近和过远的惩罚
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)        # 距离过近的惩罚
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)      # 距离过远的惩罚
        # 使用指数函数给出奖励
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        计算最小化脚部打滑的奖励。基于接触力和脚部速度计算
        使用接触阈值来判断脚部是否与地面接触，然后计算脚部速度并与接触条件相乘
        防止支撑相时脚部滑动，提高行走的稳定性
        
        返回:
            脚部打滑惩罚（负奖励）
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.  # 判断脚部是否接触地面
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)  # 计算脚部水平速度
        rew = torch.sqrt(foot_speed_norm)  # 使用平方根软化惩罚
        rew *= contact  # 只在接触时惩罚滑动
        return torch.sum(rew, dim=1)  # 返回惩罚值（负奖励）  

    def _reward_feet_air_time(self):
        """
        计算脚部空中时间奖励，促进更长的步幅
        通过检查脚部在空中后的首次接触来实现，空中时间被限制在最大值以进行奖励计算
        鼓励适当的摆动相时间，促进自然的步态
        
        返回:
            脚部空中时间奖励
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.  # 判断脚部是否接触地面
        stance_mask = self._get_gait_phase()  # 获取步态相位掩码
        # 综合真实接触、步态掩码和上次接触状态
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact  # 更新上次接触状态
        # 检查是否为首次接触（空中时间>0且当前接触）
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt  # 累积空中时间
        # 限制空中时间在合理范围内并计算奖励
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt  # 重置接触脚的空中时间
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        基于脚部接触数量与步态相位的对齐情况计算奖励
        根据脚部接触是否与预期步态相位匹配来给出奖励或惩罚
        促进正确的步态模式，左右脚交替支撑
        
        返回:
            脚部接触模式奖励
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.  # 获取真实接触状态
        stance_mask = self._get_gait_phase()  # 获取步态相位掩码
        # 当真实接触与步态相位匹配时给予正奖励，不匹配时给予负奖励
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)  # 返回平均奖励

    def _reward_orientation(self):
        """
        计算保持基座平均方向的奖励
        使用基座欧拉角和投影重力向量来惩罚从期望基座方向的偏离
        防止机器人倾斜和翻倒，保持直立行走
        
        返回:
            基座方向奖励
        """
        # 基于欧拉角的方向奖励（惩罚俏仰和横滚）
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        # 基于投影重力的方向奖励
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2  # 两个指标的平均值

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        计算精确跟踪线性和角速度命令的奖励
        惩罚从指定线性和角速度目标的偏离
        促进机器人精确按照命令运动
        
        返回:
            综合速度跟踪奖励
        """
        # 跟踪线性速度命令（xy轴）
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)  # 指数奖励函数

        # 跟踪角速度命令（偏航角）
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)  # 指数奖励函数

        # 线性误差惩罚
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        根据机器人相对于命令速度的实际速度来奖励或惩罚
        检查机器人是否移动过慢、过快或在期望速度，以及移动方向是否与命令匹配
        促进机器人遵循速度命令，防止过快或过慢的运动
        
        返回:
            速度控制奖励
        """
        # 计算速度和命令的绝对值用于比较
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # 定义期望范围的速度标准
        speed_too_low = absolute_speed < 0.5 * absolute_command     # 速度过慢
        speed_too_high = absolute_speed > 1.2 * absolute_command    # 速度过快
        speed_desired = ~(speed_too_low | speed_too_high)           # 速度合适

        # 检查速度和命令方向是否不匹配
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # 初始化奖励张量
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # 根据条件分配奖励
        reward[speed_too_low] = -1.0    # 速度过慢时惩罚
        reward[speed_too_high] = 0.     # 速度过快时中性
        reward[speed_desired] = 1.2     # 速度合适时奖励
        reward[sign_mismatch] = -2.0    # 方向不匹配时严重惩罚
        
        # 只在有明显命令时应用奖励
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        惩罚机器人关节使用高扭矩。通过最小化电机所需的力来鼓励高效的运动
        降低能量消耗，减少电机发热和磨损
        
        返回:
            扭矩惩罚（负奖励）
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        惩罚机器人自由度（DOF）的高速度。鼓励更平滑和更可控的运动
        防止关节过度运动，保持平稳的动作
        
        返回:
            关节速度惩罚（负奖励）
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        惩罚机器人自由度（DOF）的高加速度。这对于确保平滑和稳定的运动很重要，
        减少机器人机械部件的磨损
        
        返回:
            关节加速度惩罚（负奖励）
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        惩罚机器人与环境的碰撞，特别关注选定的身体部位
        鼓励机器人避免与物体或表面的不必要接触
        
        返回:
            碰撞惩罚（负奖励）
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        通过惩罚连续动作之间的大差异来鼓励机器人动作的平滑性
        这对于实现流畅运动和减少机械应力很重要
        
        返回:
            动作平滑性惩罚（负奖励）
        """
        # 第一阶差分（相邻动作的差异）
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        # 第二阶差分（加速度差异）
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        # 动作幅度惩罚
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
