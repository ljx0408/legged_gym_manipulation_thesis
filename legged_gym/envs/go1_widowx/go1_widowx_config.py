# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1WidowXRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 18     # 12腿 + 6臂
        num_observations = 72  # 明确告诉程序只要 72 维

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False  # 强制关闭地形扫描，防止产生额外的 187 维数据

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # === 四条腿 (12个关节) ===
            'FL_hip_joint': 0.1,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.1,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.1 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.1,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            # === 机械臂 (6个关节) ===
            # 名字来源 
            "widow_waist": 0.0,
            "widow_shoulder": 0.0,
            "widow_elbow": 0.0,
            "widow_forearm_roll": 0.0,
            "widow_wrist_angle": 0.0,
            "widow_wrist_rotate": 0.0,
            # 'widow_left_finger': 0,
            # 'widow_right_finger': 0
            
        }

# === 添加到 Go1WidowXRoughCfg 类中 ===
    
    class goal_ee:
        # 目标生成的命令模式：'sphere' (球坐标) 或 'cart' (笛卡尔坐标)
        command_mode = 'sphere' 
        
        # 目标保持时间：每隔多久换一个目标？(秒)
        # 这里设置为 1 到 3 秒之间随机
        traj_time = [1.0, 3.0] 
        
        class ranges:
            # === 目标生成范围 (球坐标) ===
            # l: 距离 (半径) [米]
            # p: 俯仰角 (Pitch) [弧度]
            # y: 偏航角 (Yaw) [弧度]
            
            # 初始/最终范围 (论文里有课程学习，我们先设一个固定的简单范围)
            # 距离：0.3米 到 0.6米
            final_pos_l = [0.3, 0.6] 
            # 俯仰：上下各 30度左右 (-0.5 到 0.5)
            final_pos_p = [-0.5, 0.5]
            # 偏航：左右各 45度左右 (-0.8 到 0.8)
            final_pos_y = [-0.8, 0.8]

            # 目标姿态的扰动范围 (Roll, Pitch, Yaw)
            # 暂时设为 0，让目标姿态保持水平，降低难度
            final_delta_orn = [[0, 0], [0, 0], [0, 0]]
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'joint': 50, 'widow': 5}  # [N*m/rad] kp
        damping = {'joint': 1, 'widow': 0.5}     # [N*m*s/rad] kd
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_widowx/urdf/go1_widowx.urdf"
        name = "go1_widowx"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand):
        # 确保鲁棒性加的随机质量
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.35
        max_contact_force = 500.
        only_positive_rewards = True
        # 设为 0.25 意味着：如果你偏离了 0.25米，你的得分就只有 e^-1 = 0.36分了。
        # 这个值越小，这就要求机器人抓得越准。
        tracking_ee_sigma = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            # === A. 核心生存奖励 (先站起来，别滚！) ===
            termination = -200.0  # 摔倒（触发reset）的重罚！告诉它摔倒是最大的错误。
            tracking_lin_vel = 1.0  # 听话走路（跟随速度指令）。这是行走的核心动力。
            tracking_ang_vel = 0.5  # 听话转弯。
            
            # === B. 姿态惩罚 (解决打滚问题的关键) ===
            # 惩罚身体沿 x/y 轴的倾斜角速度。让它保持背部水平，不要乱扭。
            ang_vel_xy = -0.05    
            # 惩罚垂直方向的速度。让它不要跳，也不要在这方向剧烈震荡。
            lin_vel_z = -2.0      
            
            # === C. 机械臂奖励 (你的新功能) ===
            # 权重 0.5 - 0.8 比较合适。
            # 如果太高(>1.0)，它会为了伸手而牺牲站立平衡；如果太低(<0.1)，它懒得伸手。
            tracking_ee_sphere = 0.5 
            
            # === D. 正则化/平滑项 (防止抽搐) ===
            # 惩罚力矩：让它省点力气，不要电机过热。
            torques = -0.00001    
            # 惩罚动作幅度：防止它腿和手甩得太猛。
            dof_acc = -2.5e-7
            # 惩罚关节速度：让动作慢一点，稳一点。
            dof_vel = -0.0
            # 碰撞惩罚：如果小腿或大腿撞地，扣分。
            collision = -1.0      
            # 动作连贯性：惩罚动作变化太快（抖动）。
            action_rate = -0.01   
            
            # === E. 那些暂时不用的 ===
            # 如果你没写对应的函数，设为 0 或者注释掉
            # feet_air_time = 1.0 (如果你想让它步态更好看，以后可以加)

class Go1WidowXRoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_widowx'
        load_run = -1
        
        max_iterations = 10000 # 你现在的设置就挺好
