from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go1_widowx_config import Go1WidowXRoughCfg


class Go1WidowX(LeggedRobot):
    cfg : Go1WidowXRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # 如果后面加了末端目标(Goal EE)，会在这里添加重置目标的逻辑
        # 暂时留空即可

    def _init_buffers(self):
        """
        【核心修改】初始化 Buffer 并手动解析 PD 参数字典
        """
        # 1. 让父类先干脏活累活（初始化 dof_pos, dof_vel 等）
        super()._init_buffers()
        
        # 2. 初始化 P (刚度) 和 D (阻尼) 的张量
        # device=self.device 确保这些张量在 GPU 上
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 3. 遍历所有关节，去 Config 的字典里“查表”
        # self.dof_names 是从 URDF 文件里自动读出来的关节名字列表
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            
            # 逻辑：只要关节名字包含字典里的 key，就应用对应的参数
            # 例如："FL_hip_joint" 包含了 "hip"，所以应用 stiffness['hip']
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            
            if not found:
                print(f"⚠️ [Warning] 关节 {name} 没有在 Config 里定义 PD 参数，默认设为 0")
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.

        # === 【建议新增】显式定义命令缩放比例 ===
        # 确保它和你的 obs_commands (3维) 对应
        # 需要用到 self.cfg.normalization.obs_scales，所以父类初始化要在前面
        self.commands_scale = torch.tensor(
            [self.cfg.normalization.obs_scales.lin_vel, 
             self.cfg.normalization.obs_scales.lin_vel, 
             self.cfg.normalization.obs_scales.ang_vel], 
            device=self.device, 
            requires_grad=False,
            )

    
    def compute_observations(self):
        """
        核心修改：构建符合论文要求的 72 维观测向量。
        包含：基座姿态(2)+角速度(3)+关节位置(18)+关节速度(18)+上一次动作(18)+指令(3)+目标(6)+接触(4)
        """
        # 1. 基座朝向 (Roll, Pitch) - [维度: 2]
        # 论文使用欧拉角替代重力投影，更适合Sim2Real
        base_rpy = get_euler_xyz(self.base_quat)
        obs_base_orn = torch.stack(base_rpy[:2], dim=-1)

        # 2. 基座角速度 - [维度: 3]
        obs_base_ang_vel = self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel

        # 3. 关节位置 (18个关节) - [维度: 18]
        # 减去默认站立姿态，计算偏差值
        obs_dof_pos = (self.dof_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos

        # 4. 关节速度 (18个关节) - [维度: 18]
        obs_dof_vel = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel

        # 5. 上一次的动作 - [维度: 18]
        # 帮助网络感知延迟和连贯性
        obs_last_actions = self.actions

        # 6. 速度指令 (x, y, yaw) - [维度: 3]
        # 告诉机器人往哪走
        obs_commands = self.commands[:, :3] * self.commands_scale

        # 7. 末端目标 (Goal) - [维度: 6]
        # 【重要】论文必须项。虽然现在还没写生成逻辑，先用0占位，保证网络结构正确。
        batch_size = self.num_envs
        obs_ee_goal = torch.zeros(batch_size, 6, device=self.device) 

        # 8. 足底接触力 - [维度: 4]
        # 论文提及的 contact indicators，帮助感知是否踩实
        obs_foot_contacts = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 0.1).float()

        # === 拼接数据 (总维度 72) ===
        self.obs_buf = torch.cat((
            obs_base_orn,       # 2
            obs_base_ang_vel,   # 3
            obs_dof_pos,        # 18
            obs_dof_vel,        # 18
            obs_last_actions,   # 18
            obs_commands,       # 3
            obs_ee_goal,        # 6
            obs_foot_contacts   # 4
        ), dim=-1)

        # 同步特权观测（暂时和普通观测一样）
        self.privileged_obs_buf = self.obs_buf
        
        return self.obs_buf
    def _compute_torques(self, actions):
        """
        【核心修改】计算电机力矩
        使用我们在 _init_buffers 里解析好的 p_gains 和 d_gains
        """
        # 1. 动作缩放：神经网络输出通常在 [-1, 1]，需要缩放到实际弧度（比如 * 0.5）
        actions_scaled = actions * self.cfg.control.action_scale
        
        # 2. 获取控制模式（通常是 'P'）
        control_type = self.cfg.control.control_type
        
        # 3. PD 控制公式
        # Torque = Kp * (目标位置 - 当前位置) - Kd * 当前速度
        if control_type == "P":
            # self.default_dof_pos 是机器人的初始站立姿态
            target_pos = actions_scaled + self.default_dof_pos
            torques = self.p_gains * (target_pos - self.dof_pos) - self.d_gains * self.dof_vel
            
        elif control_type == "V":
            # 速度控制 (一般不用)
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            
        elif control_type == "T":
            # 力矩控制 (Sim2Real 高级用法)
            torques = actions_scaled
            
        else:
            raise NameError(f"Unknown controller type: {control_type}")
            
        # 4. 限制力矩范围，防止数值爆炸 (Clip)
        return torch.clip(torques, -self.torque_limits, self.torque_limits) 