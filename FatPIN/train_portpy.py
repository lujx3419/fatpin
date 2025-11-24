#!/usr/bin/env python3
"""
PortPy版本的FatPIN训练代码
重构自train.py，替换MATLAB依赖为PortPy


# 使用自动生成的实验ID（时间戳）
python3 train_portpy.py

# 指定自定义实验ID
python3 train_portpy.py --exp-id exp_001

# 带描述的实验ID
python3 train_portpy.py --exp-id exp_lr1e4_batch64
"""

# 启用GPU加速的qpth求解器（在导入之前设置，确保优化器使用GPU）
import os
os.environ['PORTPY_SOLVER'] = 'qpth'  # 使用qpth GPU求解器
os.environ.setdefault('PORTPY_QPTH_MODE', 'standard')  # 'standard' 或 'slack'

# 验证CUDA可用性
import torch
if torch.cuda.is_available():
    print(f"[训练] CUDA可用，设备: {torch.cuda.get_device_name(0)}")
    print(f"[训练] GPU求解器已启用 (PORTPY_SOLVER=qpth)")
else:
    print("[警告] CUDA不可用，将使用CPU求解器")

import random
import numpy as np
import math
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical, Normal  
import time
import shutil
import scipy.io as scio
import skfda
import argparse
from datetime import datetime

# 导入PortPy函数
from portpy_functions import step_fda_multipatient, reset_fda_multipatient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 强化学习参数（DDPG）
gamma = 0.99
actor_lr = 1e-4
critic_lr = 1e-3
# 降低batch_size：每个患者3步，4个患者=12步，需要batch_size <= 12才能在第一轮更新
# 设置为16，这样至少需要2个患者（6步）就能开始更新，但不会太小导致训练不稳定
batch_size = 16
Episode = 30
tau = 5e-3  # 软更新系数
max_steps_per_patient = 50
replay_capacity = 100000
exploration_noise_std_init = 0.1
exploration_noise_std_min = 0.02
exploration_noise_decay = 0.995


def composite_approximator(f_beta, f_xt, a, b, n):
    """复合近似器 - 保持原始实现"""
    def f_x(t):
        x = t
        return eval(f_beta) * (f_xt(x).ravel()) 

    # 使用复合辛普森公式计算积分
    integ_approx = ((b-a))/6 * (f_x(a) + 4*f_x(((a+b)/2)) + f_x(b))
    return integ_approx


def intergration_form_fourier(functional_data,
                            beta_basis=None, 
                            num_fd_basis=None,  # fd基函数数量
                            num_beta_basis=None,  # beta基函数数量
                            domin_range=None):
    """傅里叶积分形式 - 保持原始实现"""
    
    # 数据清理和验证
    functional_data = np.array(functional_data).flatten()
    domin_range = np.array(domin_range).flatten()
    
    # 检查并清理无效值（inf, NaN）
    if np.any(np.isinf(functional_data)) or np.any(np.isnan(functional_data)):
        print(f"[警告] functional_data包含无效值，进行清理...")
        functional_data = np.nan_to_num(functional_data, nan=0.0, posinf=np.max(functional_data[~np.isinf(functional_data)]) if np.any(~np.isinf(functional_data)) else 0.0, neginf=0.0)
    
    # 检查domin_range的有效性
    if np.any(np.isinf(domin_range)) or np.any(np.isnan(domin_range)):
        print(f"[警告] domin_range包含无效值，使用默认范围...")
        domin_range = np.linspace(0, 2.0, len(functional_data))
    
    # 确保domin_range是单调递增的且无重复值
    if len(np.unique(domin_range)) < len(domin_range):
        print(f"[警告] domin_range有重复值，进行去重和插值...")
        unique_range, unique_indices = np.unique(domin_range, return_index=True)
        if len(unique_range) < 2:
            # 如果去重后点数太少，使用线性范围
            domin_range = np.linspace(0, 2.0, len(functional_data))
        else:
            # 对functional_data进行插值到去重后的范围
            from scipy.interpolate import interp1d
            unique_data = functional_data[unique_indices]
            # 确保unique_range是递增的
            sort_idx = np.argsort(unique_range)
            unique_range = unique_range[sort_idx]
            unique_data = unique_data[sort_idx]
            interp_func = interp1d(unique_range, unique_data, kind='linear', fill_value='extrapolate', assume_sorted=True)
            # 创建新的均匀分布的范围
            new_range = np.linspace(unique_range[0], unique_range[-1], len(functional_data))
            functional_data = interp_func(new_range)
            domin_range = new_range
    
    # 确保domin_range是递增的
    if not np.all(np.diff(domin_range) > 0):
        print(f"[警告] domin_range不是严格递增，进行排序...")
        sort_indices = np.argsort(domin_range)
        domin_range = domin_range[sort_indices]
        functional_data = functional_data[sort_indices]
    
    # 验证数据长度匹配
    if len(functional_data) != len(domin_range):
        min_len = min(len(functional_data), len(domin_range))
        functional_data = functional_data[:min_len]
        domin_range = domin_range[:min_len]
        print(f"[警告] functional_data和domin_range长度不匹配，截断到 {min_len}")
    
    #### 设置 x_i(s) 形式 ####
    grid_point = np.array(domin_range).squeeze()             
    data_matrix = np.array(functional_data)[np.newaxis, :]
    
    # 再次检查数据有效性
    if np.any(np.isinf(data_matrix)) or np.any(np.isnan(data_matrix)):
        raise ValueError(f"数据清理后仍有无效值: inf={np.any(np.isinf(data_matrix))}, nan={np.any(np.isnan(data_matrix))}")
    
    # 使用新的BSplineBasis类
    basis = skfda.representation.basis.BSplineBasis(n_basis=num_fd_basis)
    fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_point)
    X_basis = fd.to_basis(basis)
    
    #### 设置 beta_(s) ####
    beta_basis_form = []
    beta_basis_form.append('1')

    for m in range(1, (num_beta_basis-1)//2+1):
        beta_basis_form.append(('np.sin(2*np.pi*x'+'*'+str(m)+'/'+str(domin_range[-1])+')'))
        beta_basis_form.append(('np.cos(2*np.pi*x'+'*'+str(m)+'/'+str(domin_range[-1])+')'))

    #### 获取近似值 ####
    integ_approximations = []
    for m in range(len(beta_basis_form)):
        # 计算 Φm(t)X(t)
        form_approximated = str(beta_basis_form[m])
        final_func = form_approximated

        integ_approximations.append(composite_approximator(
            f_beta=final_func, f_xt=X_basis, 
            a=domin_range[0], b=domin_range[-1], n=5000))

    return integ_approximations


def func_to_state(functional_data, domin_range):
    """功能数据转状态 - 调用原始的傅里叶积分方法"""
    num_fd_basis = 40
    num_beta_basis = 5 
    
    # 确保functional_data是2维数组
    if functional_data.ndim == 1:
        raise ValueError(f"functional_data是1维数组，形状: {functional_data.shape}，无法处理")
    
    # 验证输入数据
    functional_data = np.array(functional_data)
    domin_range = np.array(domin_range).flatten()
    
    # 检查数据有效性
    if np.any(np.isinf(functional_data)) or np.any(np.isnan(functional_data)):
        print(f"[警告] func_to_state: functional_data包含无效值，进行清理...")
        functional_data = np.nan_to_num(functional_data, nan=0.0, posinf=100.0, neginf=0.0)
    
    num_organs = functional_data.shape[0]
    state = np.zeros((num_organs, num_beta_basis))
    
    for i in range(num_organs):
        # 获取第i个器官的功能数据
        if functional_data.ndim == 2:
            organ_data = functional_data[i, :].copy()
        else:
            organ_data = np.array(functional_data[i]).flatten()
        
        # 检查单个器官数据
        if np.any(np.isinf(organ_data)) or np.any(np.isnan(organ_data)):
            print(f"[警告] 器官 {i} 的数据包含无效值，使用0替换")
            organ_data = np.nan_to_num(organ_data, nan=0.0, posinf=100.0, neginf=0.0)
        
        # 调用原始的intergration_form_fourier函数
        try:
            result = intergration_form_fourier(
                organ_data, 
                num_fd_basis=num_fd_basis, 
                num_beta_basis=num_beta_basis, 
                domin_range=domin_range)
            # 将列表转换为数组并展平
            result_array = np.array(result).flatten()
            
            # 检查结果有效性
            if np.any(np.isinf(result_array)) or np.any(np.isnan(result_array)):
                print(f"[警告] 器官 {i} 的积分结果包含无效值，使用0替换")
                result_array = np.nan_to_num(result_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            state[i, :] = result_array
        except Exception as e:
            print(f"[错误] 器官 {i} 的傅里叶积分失败: {e}，使用零向量")
            state[i, :] = 0.0

    return state


class TP():
    """强化学习环境 - 重构为PortPy版本"""
    
    def __init__(self, num_of_organ):
        # 连续动作区间 [low, high]，每个器官一个缩放系数
        self.action_low = 0.95
        self.action_high = 1.05
        self.num_of_organs = num_of_organ
        self.observation_space = np.random.randn(num_of_organ, 5)  
        # 训练控制
        self.min_steps_before_done = 3
        self._episode_step = 0
        # Action范围：从±5%扩大到±20%，允许更大的探索和调整空间
        self.action_low = 0.8
        self.action_high = 1.2

    def step(self, file, action, dose, V18, D95):
        """执行一步强化学习 - 使用PortPy替代MATLAB"""
        
        # 维度已确认正常，移除详细调试输出
        
        # 调用PortPy版本的step函数
        result = step_fda_multipatient(file, action, dose)

        f_data = np.array(result[0])                      
        d_range = np.array(result[1]).squeeze()   
        dose_ = np.array(result[2])                       
        V18_ = np.array(result[3])
        D95_ = np.array(result[4]).reshape(1, -1)
        A_target = float(result[5])
        A_oar = float(result[6])

        V18 = np.array(V18)
        D95 = np.array(D95)

        # 论文式连续奖励: r = ΔD95 + λ*(A_Target - \bar{A}_OAR)
        # 为避免面积主导奖励，将面积做无量纲归一化：除以 (maxDose * 100%)，并乘以缩放系数 λ
        delta_D95 = float(np.mean(D95_) - np.mean(D95))
        max_dose = float(np.max(d_range)) if d_range is not None and len(d_range) > 0 else 50.0
        denom = max(max_dose * 100.0, 1e-6)
        a_t_norm = float(A_target) / denom
        a_o_norm = float(A_oar) / denom
        area_term = 10.0 * (a_t_norm - a_o_norm)
        reward = delta_D95 + area_term
        
        # 调试信息：打印reward组成，帮助理解为什么reward在固定值之间跳
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
        
        if self._step_count % 10 == 1:  # 每10步打印一次详细信息
            print(f"[Reward调试] Step {self._step_count}:")
            print(f"  D95变化: {np.mean(D95):.4f} -> {np.mean(D95_):.4f}, ΔD95 = {delta_D95:.4f}")
            print(f"  A_target = {A_target:.4f}, A_oar = {A_oar:.4f}, (A_t - A_o) = {(A_target - A_oar):.4f}")
            print(f"  Reward = {reward:.4f}")
            print(f"  剂量范围: {d_range[0]:.3f} - {d_range[-1]:.3f} Gy (可能归一化)")
            print(f"  Action范围: [{np.min(action):.3f}, {np.max(action):.3f}]")
            
        done = False

        # 计步
        self._episode_step += 1

        # 结束条件：
        # - 使用新状态 V18_ 检查OAR体积阈值（0.5表示50%）
        # - 若D95下降（相对于上一步）则可视为不良，也可提前终止
        # - 前 min_steps_before_done 步一律不结束，保证至少多步交互
        done = False
        if self._episode_step >= self.min_steps_before_done:
            v18_check = np.any(V18_[:] >= 0.5)
            d95_check = np.any(D95_[:] < D95[:])
            if v18_check or d95_check:
                done = True
                # 调试：打印done条件触发原因
                if self._step_count % 10 == 1 or self._step_count <= 5:
                    print(f"[Done条件] Step {self._episode_step}: V18>=0.5={v18_check}, D95_<D95={d95_check}")
                    print(f"  V18_: {V18_[:5] if len(V18_) >= 5 else V18_}")
                    print(f"  D95_: {D95_[:5] if len(D95_) >= 5 else D95_}, D95: {D95[:5] if len(D95) >= 5 else D95}")

        info = {}

        return f_data, d_range, reward, dose_, done, V18_, D95_

    def reset(self, file, num_of_organ):
        """重置环境 - 使用PortPy替代MATLAB"""
        
        dose = [50] * num_of_organ

        # 调用PortPy版本的reset函数
        # 重置计步器
        self._episode_step = 0

        result = reset_fda_multipatient(file, dose)
        
        functional_data = np.array(result[0])                              
        domin_range = np.array(result[1]).squeeze()  # 直接使用numpy，避免torch.FloatTensor的警告
        # 修复：直接转换为numpy数组，避免先转torch tensor的性能问题
        D95 = np.array(result[3]) if isinstance(result[3], (list, tuple)) else np.array([result[3]])
        V18 = np.array(result[2]).squeeze()
        # 使用reset_fda_multipatient返回的实际剂量（维度已验证）
        dose_actual = result[4] if len(result) > 4 else dose

        return functional_data, domin_range, dose_actual, V18, D95

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s_, d):
        idx = self.ptr
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r
        self.next_state[idx] = s_
        self.done[idx] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idxs], dtype=torch.float32, device=device),
            torch.as_tensor(self.action[idxs], dtype=torch.float32, device=device),
            torch.as_tensor(self.reward[idxs], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_state[idxs], dtype=torch.float32, device=device),
            torch.as_tensor(self.done[idxs], dtype=torch.float32, device=device),
        )


class DDPGActor(nn.Module):
    """高斯策略Actor：器官级嵌入 X(M,K)->h(M)，再经 MLP 产出均值与方差"""
    def __init__(self, state_dim: int, action_dim: int, action_low: float, action_high: float):
        super().__init__()
        # 从维度推断器官数与每器官特征数 K
        assert state_dim % action_dim == 0, "state_dim 必须能被 action_dim 整除以形成 (M,K)"
        self.num_organs = action_dim  # M
        self.per_organ_k = state_dim // action_dim  # K

        # 器官级线性映射：每个器官 K 维 -> 1 标量，然后激活得到 h ∈ R^M
        self.organ_embed = nn.Linear(self.per_organ_k, 1)
        self.act = nn.ReLU()

        # h -> z: 两个全连接层
        self.fc1 = nn.Linear(self.num_organs, 256)
        self.fc2 = nn.Linear(256, 256)

        # z -> 均值和方差
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer("log_std_min", torch.tensor(-10.0, dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(2.0, dtype=torch.float32))

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> tuple:
        """
        返回: (action, log_prob)
        state 形状: [B, M*K]，内部重塑为 [B, M, K] 计算 h
        """
        B = state.shape[0]
        x = state.view(B, self.num_organs, self.per_organ_k)
        # 器官级线性映射并激活: [B,M,K] -> [B,M,1] -> [B,M]
        h = self.organ_embed(x).squeeze(-1)
        h = self.act(h)

        # h -> z
        z = self.act(self.fc1(h))
        z = self.act(self.fc2(z))

        # z -> 均值与 log_std
        mean = self.mean_head(z)
        log_std = self.log_std_head(z)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        if deterministic:
            action = mean
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        action = torch.clamp(action, self.action_low, self.action_high)
        return action, log_prob


class DDPGCritic(nn.Module):
    """Critic：内部将 X(M,K) 映射为 h(M)，与 a(M) 拼接后评估 Q"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        assert state_dim % action_dim == 0, "state_dim 必须能被 action_dim 整除以形成 (M,K)"
        self.num_organs = action_dim  # M
        self.per_organ_k = state_dim // action_dim  # K

        # 独立的器官级嵌入（不与Actor共享参数）
        self.organ_embed = nn.Linear(self.per_organ_k, 1)
        self.act = nn.ReLU()

        # 输入为 [h, a]，即 2M 维
        self.net = nn.Sequential(
            nn.Linear(self.num_organs * 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        x = state.view(B, self.num_organs, self.per_organ_k)
        h = self.organ_embed(x).squeeze(-1)
        h = self.act(h)
        qa_in = torch.cat([h, action], dim=-1)
        return self.net(qa_in)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data)


def main():
    """主训练函数 - DDPG 版本（4 网络 + 软更新）"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FatPIN Training with PortPy')
    parser.add_argument('--exp-id', type=str, default=None, 
                       help='Experiment ID. If not specified, use timestamp like "exp_YYYYMMDD_HHMMSS"')
    args = parser.parse_args()
    
    # 确定实验ID
    if args.exp_id is None:
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_id = args.exp_id
    
    # 创建实验专用结果目录
    base_result_dir = f'./result/{exp_id}'
    os.makedirs(f'{base_result_dir}/train_info', exist_ok=True)
    os.makedirs(f'{base_result_dir}/actor', exist_ok=True)
    os.makedirs(f'{base_result_dir}/critic', exist_ok=True)
    
    print(f'Experiment ID: {exp_id}')
    print(f'Results will be saved to: {base_result_dir}')

    # 动态获取实际器官数量（从第一个患者加载数据）
    print('Loading first patient to determine number of organs...')
    from portpy_structures import load_patient_data
    # 从配置文件读取优化体素分辨率（用于加速）
    try:
        from config import OPT_VOXEL_RESOLUTION_MM
        opt_vox_res = OPT_VOXEL_RESOLUTION_MM
        print(f'使用优化体素分辨率: {opt_vox_res}mm（加速优化）')
    except ImportError:
        opt_vox_res = None  # 如果没有config，使用默认
        print('使用默认优化体素分辨率')
    
    # 遍历所有患者，找到最大的有约束器官数量，确保网络维度能容纳所有患者
    file_list = ['Lung_Patient_2', 'Lung_Patient_3', 'Lung_Patient_4', 'Lung_Patient_5']
    max_constraint_organs = 0
    patient_constraint_counts = {}
    
    print('扫描所有患者，确定最大有约束器官数量...')
    for patient_file in file_list:
        cst, _, _, _ = load_patient_data(patient_file, opt_vox_res_mm=opt_vox_res)
        cst_Inx = cst.get_constraint_indices()
        num_constraint_organs = len(cst_Inx)
        patient_constraint_counts[patient_file] = num_constraint_organs
        max_constraint_organs = max(max_constraint_organs, num_constraint_organs)
        print(f'  {patient_file}: {num_constraint_organs} 个有约束的器官')
    
    print(f'\n最大有约束器官数量: {max_constraint_organs} (将用于网络维度)')
    
    # 使用最大维度创建环境（确保能容纳所有患者）
    env = TP(num_of_organ=max_constraint_organs)

    # 维度定义（使用最大维度）
    per_organ_state_dim = env.observation_space.shape[1]
    state_dim = max_constraint_organs * per_organ_state_dim  # 展平后的状态维度
    action_dim = max_constraint_organs  # 每个有约束的器官一个连续动作缩放系数
    
    print(f'网络维度: state_dim={state_dim}, action_dim={action_dim}, per_organ_state_dim={per_organ_state_dim}')

    # 网络
    print(f'\n[网络] 使用设备: {device}')
    actor = DDPGActor(state_dim, action_dim, env.action_low, env.action_high).to(device)
    critic = DDPGCritic(state_dim, action_dim).to(device)
    actor_target = DDPGActor(state_dim, action_dim, env.action_low, env.action_high).to(device)
    critic_target = DDPGCritic(state_dim, action_dim).to(device)
    hard_update(actor_target, actor)
    hard_update(critic_target, critic)
    
    # 验证网络是否在GPU上
    if device.type == 'cuda':
        print(f'[网络] Actor参数在GPU: {next(actor.parameters()).is_cuda}')
        print(f'[网络] Critic参数在GPU: {next(critic.parameters()).is_cuda}')
    else:
        print('[网络] 警告: 网络在CPU上运行')

    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    mse_loss = nn.MSELoss()

    # 经验回放
    buffer = ReplayBuffer(replay_capacity, state_dim, action_dim)
    noise_std = exploration_noise_std_init

    # 训练轮次
    for episode in range(Episode):
        file_list = ['Lung_Patient_2', 'Lung_Patient_3', 'Lung_Patient_4', 'Lung_Patient_5']  # 患者列表（PortPy目录）
        num_of_file = 0
        for file in file_list:
            num_of_file += 1
            print('training in the file', file)
            
            # 获取当前患者的有约束器官数量
            # 注意：这里必须与reset_fda_multipatient内部加载的数据保持一致
            # reset_fda_multipatient内部会重新加载数据，所以这里只是用来初始化dose
            cst_preload, _, _, _ = load_patient_data(file, opt_vox_res_mm=opt_vox_res)
            preload_constraint_organs = len(cst_preload.get_constraint_indices())
            preload_struct_names = list(cst_preload.structures.keys())
            preload_constraint_names = [preload_struct_names[i] for i in cst_preload.get_constraint_indices()]
            
            print(f"[维度检查] 患者 {file}: 预加载得到的约束数量 = {preload_constraint_organs}")
            print(f"[维度检查] 有约束的结构: {preload_constraint_names}")
            
            # 初始化环境（使用预加载的约束数量来初始化dose）
            # reset_fda_multipatient内部会重新加载并验证，如果不匹配会抛出错误
            f_data, d_range, dose, V18, D95 = env.reset(file, num_of_organ=preload_constraint_organs)
            
            # 验证返回的dose维度
            dose_len = len(dose) if hasattr(dose, '__len__') else 1
            if dose_len != preload_constraint_organs:
                raise ValueError(
                    f"维度不匹配！患者 {file}:\n"
                    f"  预加载约束数量: {preload_constraint_organs}\n"
                    f"  reset返回的dose维度: {dose_len}\n"
                    f"  这说明reset_fda_multipatient内部加载的数据与外部不一致！"
                )
            print(f"[维度检查] reset返回的dose维度 = {dose_len}，匹配 ✓")  

            # 验证维度匹配
            actual_num_organs = f_data.shape[0]
            if actual_num_organs != preload_constraint_organs:
                raise ValueError(f"患者 {file} 的结构数量 ({actual_num_organs}) 与期望数量 ({preload_constraint_organs}) 不匹配！")
            
            state_mat = func_to_state(f_data, d_range)
            state_raw = state_mat.reshape(-1)  # 展平到 [preload_constraint_organs * per_organ_state_dim]
            
            # 如果当前患者的维度小于最大维度，需要padding到state_dim
            if len(state_raw) < state_dim:
                # Padding: 用0填充到state_dim
                state = np.pad(state_raw, (0, state_dim - len(state_raw)), mode='constant', constant_values=0.0)
                print(f"[维度调整] 患者 {file} 的state维度从 {len(state_raw)} padding到 {state_dim}")
            else:
                state = state_raw
            
            total_reward = 0.0

            print("Initialization completed, Patient {0}, Round No. {1} begins".format(file, episode+1))
            
            # 调试：记录初始状态和剂量，用于检查是否每次round都相同
            state_hash = hash(tuple(state[:10])) if len(state) >= 10 else hash(tuple(state))
            dose_str = str(dose[:5]) if len(dose) >= 5 else str(dose)
            print(f"[调试-Round开始] Episode {episode+1}, Patient {file}, Step 0 (初始)")
            print(f"  初始state哈希(前10个元素): {state_hash}, 初始state前5值: {state[:5]}")
            print(f"  初始dose: {dose_str}, V18: {V18}, D95: {D95}")
            
            for i_train in range(max_steps_per_patient):  
                env.render()

                # 选择动作（高斯策略：从均值和方差得到的高斯分布采样）
                actor.eval()
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor, _ = actor(s_t, deterministic=False)  # 从高斯分布采样
                    action_full = action_tensor.squeeze(0).cpu().numpy()
                    # action_full是[action_dim]形状，但当前患者只需要前preload_constraint_organs个
                    if len(action_full.shape) == 0:
                        action_full = action_full.reshape(1)
                    elif len(action_full.shape) > 1:
                        action_full = action_full.flatten()
                    
                    # 截取当前患者需要的action维度
                    if len(action_full) < preload_constraint_organs:
                        raise ValueError(
                            f"action_full维度 ({len(action_full)}) < preload_constraint_organs ({preload_constraint_organs})！\n"
                            f"  action_dim (网络输出维度) = {action_dim}\n"
                            f"  action_full实际长度 = {len(action_full)}\n"
                            f"  需要截取的长度 = {preload_constraint_organs}"
                        )
                    action = action_full[:preload_constraint_organs]
                actor.train()

                # 与PortPy交互（使用实际器官数量）
                functional_data, domin_range, reward, dose_, done, V18_, D95_ = env.step(file, action, dose, V18, D95)

                # 验证维度匹配
                actual_num_organs = functional_data.shape[0]
                if actual_num_organs != preload_constraint_organs:
                    raise ValueError(f"患者 {file} 的结构数量 ({actual_num_organs}) 与期望数量 ({preload_constraint_organs}) 不匹配！")

                # 下一个状态
                next_state_mat = func_to_state(functional_data, domin_range)
                next_state_raw = next_state_mat.reshape(-1)
                
                # Padding到state_dim（如果需要）
                if len(next_state_raw) < state_dim:
                    next_state = np.pad(next_state_raw, (0, state_dim - len(next_state_raw)), mode='constant', constant_values=0.0)
                else:
                    next_state = next_state_raw

                total_reward += float(reward)

                # 存入回放（action也需要padding到action_dim）
                action_padded = np.pad(action, (0, action_dim - len(action)), mode='constant', constant_values=0.0) if len(action) < action_dim else action
                buffer.push(state.astype(np.float32), action_padded.astype(np.float32), np.array([reward], dtype=np.float32), next_state.astype(np.float32), np.array([float(done)], dtype=np.float32))

                # 更新网络
                # 注意：如果buffer.size < batch_size，使用实际大小进行更新（但至少需要1个样本）
                if buffer.size >= 1:
                    actual_batch_size = min(batch_size, buffer.size)
                    bs, ba, br, bs_, bd = buffer.sample(actual_batch_size)
                    # Critic目标：y = r + gamma*(1-d)*Q'(s', a'(s'))
                    with torch.no_grad():
                        a_next, _ = actor_target(bs_, deterministic=True)  # 使用均值（确定性）
                        q_target_next = critic_target(bs_, a_next)
                        y = br + gamma * (1.0 - bd) * q_target_next
                    # Critic更新
                    q_val = critic(bs, ba)
                    critic_loss = mse_loss(q_val, y)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()
                    # Actor更新：最大化 Q(s, a(s)) -> 最小化 -Q
                    # 使用确定性策略（均值）来最大化Q值（DDPG风格）
                    pred_action, _ = actor(bs, deterministic=True)
                    actor_loss = -critic(bs, pred_action).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()
                    # 软更新
                    soft_update(actor_target, actor, tau)
                    soft_update(critic_target, critic, tau)
                    
                    # 调试：打印网络更新信息（检查网络是否真的在更新）
                    if i_train == 0 or (i_train < 5) or (buffer.size % 10 == 0):
                        # 获取actor网络第一层权重的均值，用于检查网络是否在更新
                        with torch.no_grad():
                            first_layer_weight = next(actor.parameters())
                            weight_mean = first_layer_weight.mean().item()
                            weight_std = first_layer_weight.std().item()
                        print(f"[网络更新] Episode {episode+1}, Patient {file}, Step {i_train+1}")
                        print(f"  Buffer size: {buffer.size}, 实际batch_size: {actual_batch_size}")
                        print(f"  Actor loss: {actor_loss.item():.6f}, Critic loss: {critic_loss.item():.6f}")
                        print(f"  Actor第一层权重: mean={weight_mean:.6f}, std={weight_std:.6f}")
                        # 检查loss是否非零，如果loss为0说明梯度可能有问题
                        if abs(actor_loss.item()) < 1e-8 and abs(critic_loss.item()) < 1e-8:
                            print(f"  [警告] Loss接近0，可能梯度有问题或数据异常")

                print('Round {0} --- Patient {1} --- Step {2} --- Reward {3:.3f}'.format(
                    episode+1, file, i_train+1, float(reward)))

                # 记录训练信息
                with open(f'{base_result_dir}/train_info/train_{episode+1}_rounds.csv', 'a', encoding='utf-8') as f:
                    # 记录剂量统计：假设index0为Target，其余为OAR
                    dose_target = float(dose_[0]) if hasattr(dose_, '__len__') and len(dose_) > 0 else float('nan')
                    dose_oar_mean = float(np.mean(dose_[1:])) if hasattr(dose_, '__len__') and len(dose_) > 1 else float('nan')
                    f.write(str(file) + ' the {}th training'.format(i_train+1)
                            + ' V18:' + str(np.array(V18_).reshape(1, -1)) + ' D95:' + str(np.array(D95_).reshape(1, -1))
                            + ' dose_target:' + str(np.array([dose_target]).reshape(1, -1))
                            + ' dose_oar_mean:' + str(np.array([dose_oar_mean]).reshape(1, -1))
                            + ' reward is ' + str(np.array(reward).reshape(1, -1))
                            + ' total reward is ' + str(np.array(total_reward).reshape(1, -1))
                            + '\n')

                # 过渡到下一步
                state = next_state
                dose = dose_
                V18 = V18_
                D95 = D95_

                if i_train % 10 == 0:
                    # 分别保存 Actor/Critic 到各自目录，便于检查
                    torch.save(actor.state_dict(), f'{base_result_dir}/actor/actor_ep{episode+1}_step{i_train+1}.pth')
                    torch.save(actor_target.state_dict(), f'{base_result_dir}/actor/actor_target_ep{episode+1}_step{i_train+1}.pth')
                    torch.save(critic.state_dict(), f'{base_result_dir}/critic/critic_ep{episode+1}_step{i_train+1}.pth')
                    torch.save(critic_target.state_dict(), f'{base_result_dir}/critic/critic_target_ep{episode+1}_step{i_train+1}.pth')
                    # 合并保存一份完整快照
                    torch.save({
                        'actor': actor.state_dict(),
                        'actor_target': actor_target.state_dict(),
                        'critic': critic.state_dict(),
                        'critic_target': critic_target.state_dict(),
                        'episode': episode,
                        'step': i_train,
                        'exp_id': exp_id,
                    }, f'{base_result_dir}/actor_critic_ddpg_{episode+1}_{i_train+1}.pth')

                if done:
                    break

            # 注意：高斯策略的探索通过网络输出的方差控制，不需要额外衰减噪声
            # 方差会通过训练自然调整


if __name__ == "__main__":
    main()
