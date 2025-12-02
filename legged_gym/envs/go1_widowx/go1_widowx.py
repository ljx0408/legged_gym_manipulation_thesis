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

# === 在 go1_widowx.py 开头添加 ===

def sphere2cart(sphere_coords):
    l, p, y = sphere_coords[:, 0], sphere_coords[:, 1], sphere_coords[:, 2]
    x = l * torch.cos(p) * torch.cos(y)
    y = l * torch.cos(p) * torch.sin(y)
    z = l * torch.sin(p)
    return torch.stack([x, y, z], dim=-1)

class Go1WidowX(LeggedRobot):
    cfg : Go1WidowXRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # 如果后面加了末端目标(Goal EE)，会在这里添加重置目标的逻辑
        # === 【新增】环境重置时，生成新目标 ===
        self._resample_ee_goal(env_ids) # 重置时必须刷新目标

    def _init_buffers(self):
        """
        【核心修改】初始化 Buffer 并手动解析 PD 参数字典
        """
        # 1. 让父类先干脏活累活（初始化 dof_pos, dof_vel 等）
        super()._init_buffers()
        # === 【新增】初始化刚体状态张量 (Rigid Body State) ===
        # 这一步是为了获取机械臂末端（手）的绝对位置，父类默认没有做这一步
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # view(num_envs, num_bodies, 13)
        # 13 包含: position(3) + orientation(4) + linear_vel(3) + angular_vel(3)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

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
        
        self.action_scale = torch.tensor(
            self.cfg.control.action_scale, 
            device=self.device, 
            dtype=torch.float
        )

# === 【新增】初始化目标相关的张量 ===
        # 1. 读取范围配置
        self.goal_ee_l_range = torch.tensor(self.cfg.goal_ee.ranges.final_pos_l, device=self.device)
        self.goal_ee_p_range = torch.tensor(self.cfg.goal_ee.ranges.final_pos_p, device=self.device)
        self.goal_ee_y_range = torch.tensor(self.cfg.goal_ee.ranges.final_pos_y, device=self.device)
        self.goal_ee_traj_time = torch.tensor(self.cfg.goal_ee.traj_time, device=self.device)

        # 2. 初始化存储张量
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        
        # 3. 马上生成第一次目标，防止开局是0
        self._resample_ee_goal(torch.arange(self.num_envs, device=self.device))

        # === 【新增】找到机械臂末端(夹爪)的索引 ===
        self.gripper_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], 
            self.actor_handles[0], 
            "wx250s/ee_gripper_link"
        )

    def _resample_ee_goal(self, env_ids):
        if len(env_ids) == 0: return

        # 1. 随机生成球坐标 [0,1] -> [min, max]
        rng = torch.rand(len(env_ids), 3, device=self.device)
        l = self.goal_ee_l_range[0] + rng[:, 0] * (self.goal_ee_l_range[1] - self.goal_ee_l_range[0])
        p = self.goal_ee_p_range[0] + rng[:, 1] * (self.goal_ee_p_range[1] - self.goal_ee_p_range[0])
        y = self.goal_ee_y_range[0] + rng[:, 2] * (self.goal_ee_y_range[1] - self.goal_ee_y_range[0])

        self.ee_goal_sphere[env_ids, 0] = l
        self.ee_goal_sphere[env_ids, 1] = p
        self.ee_goal_sphere[env_ids, 2] = y

        # 2. 转换为直角坐标 (给奖励函数和观测用)
        # 注意：这里需要加上基座偏移（机械臂不是长在地面上的，是长在狗背上的）
        # 简单起见，我们先假设目标是相对于【机械臂基座】的
        self.ee_goal_cart[env_ids] = sphere2cart(self.ee_goal_sphere[env_ids])

        # 3. 重置倒计时
        time_s = self.goal_ee_traj_time[0] + torch.rand(len(env_ids), device=self.device) * (self.goal_ee_traj_time[1] - self.goal_ee_traj_time[0])
        self.goal_timer[env_ids] = time_s / self.dt

    
    def _post_physics_step_callback(self):

        # 1. 刷新刚体状态 (告诉物理引擎：更新一下机械臂的位置数据！)
        # 如果不加这行，self.rigid_body_state 里的数据永远是静止的
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        super()._post_physics_step_callback()
        
        # 倒计时
        self.goal_timer -= 1
        # 谁的时间到了，就给谁换个新目标
        reset_ids = (self.goal_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._resample_ee_goal(reset_ids)
       
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

        # 7. 末端目标 (Goal)
        # 使用我们实时更新的直角坐标目标
        # 还需要加上目标姿态 (Config里设了final_delta_orn为0，所以这里也先用0占位或者简单处理)
        batch_size = self.num_envs
        target_pos = self.ee_goal_cart # [Batch, 3]
        target_orn = torch.zeros(batch_size, 3, device=self.device) # [Batch, 3] 暂时用0表示姿态目标

        obs_ee_goal = torch.cat([target_pos, target_orn], dim=-1) # [Batch, 6]
        
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
# === 【新增】机械臂追踪奖励函数 ===
    def _reward_tracking_ee_sphere(self):
        # 1. 获取手的位置 (世界坐标)
        ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]

        # 2. 获取目标位置 (相对坐标)
        target_pos_rel = self.ee_goal_cart
        
        # 3. 把目标转成世界坐标
        # 目标世界坐标 = 机器人基座位置 + 旋转后的相对目标
        # 注意：这里我们做一个简化，假设机器人身体是平的，直接加基座位置
        # (更严谨的做法是用四元数旋转 target_pos_rel，但先跑通再说)
        base_pos = self.root_states[:, :3]
        # 加上基座高度偏移 (机械臂是装在狗背上的，不是脚底)
        # 假设机械臂安装高度是 0.35m (init_state.pos) + 0.05m (安装座)
        # 简单起见，我们直接用 base_pos + target_pos_rel 试试
        # 更好的做法是在 compute_observations 里就把相对坐标喂给网络，奖励函数也算相对距离
        
        # --- 推荐方案：计算相对距离 ---
        # 把 ee_pos (世界) 减去 base_pos (世界) = ee_pos_rel (相对)
        ee_pos_rel = ee_pos - self.root_states[:, :3]
        
        # 然后算 ee_pos_rel 和 target_pos_rel 的距离
        distance = torch.norm(target_pos_rel - ee_pos_rel, dim=-1)

        reward = torch.exp(-distance / self.cfg.rewards.tracking_ee_sigma)
        return reward
       
    def _compute_torques(self, actions):
        """
        【核心修改】计算电机力矩
        使用我们在 _init_buffers 里解析好的 p_gains 和 d_gains
        """
        # 1. 动作缩放：神经网络输出通常在 [-1, 1]，需要缩放到实际弧度（比如 * 0.5）
        actions_scaled = actions * self.action_scale
        
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