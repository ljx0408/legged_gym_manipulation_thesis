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