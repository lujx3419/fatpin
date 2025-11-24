#!/usr/bin/env python3
"""
Usage examples:

# 1) Rollout a single trained checkpoint on one patient for N steps and plot dose trend
python3 FatPIN/plot_training_curves.py \
    --patient Lung_Patient_2 \
    --rollout-ckpt FatPIN/result/exp_005/actor/actor_ep30_step1.pth \
    --rollout-steps 30 \
    --trend-out FatPIN/result/exp_005/visualization/dose_trend_rollout_Lung_Patient_2.png

# 2) Rollout the same checkpoint and plot DVH snapshots at t=0,10,20,30
python3 FatPIN/plot_training_curves.py \
    --patient Lung_Patient_2 \
    --rollout-ckpt FatPIN/result/exp_005/actor/actor_ep30_step1.pth \
    --dvh-steps 10,20,30 \
    --dvh-out FatPIN/result/exp_005/visualization/dvh_rollout_Lung_Patient_2_ep30.png

# 3) (Fallback) Read training CSV to plot dose trend (if no checkpoint rollout)
python3 FatPIN/plot_training_curves.py \
    --exp-id exp_005 \
    --episode 30 \
    --patient Lung_Patient_2 \
    --trend-out FatPIN/result/exp_005/visualization/dose_trend_from_csv_Lung_Patient_2_ep30.png
"""
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from visualize_results import process_dvh  # reuse helper
from visualize_results import plot_dvh  # optional reference
from train_portpy import TP  # environment
from portpy_functions import reset_fda_multipatient
import torch


def parse_train_csv(csv_path: str, patient_filter: str = None):
    steps = []
    dose_ptv = []
    dose_oar_mean = []
    d95 = []
    v18_mean = []

    if not os.path.exists(csv_path):
        return steps, dose_ptv, dose_oar_mean, d95, v18_mean

    with open(csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # line example:
            # Lung_Patient_2 the 1th training V18:[[...]] D95:[[...]] dose_target:[[..]] dose_oar_mean:[[..]] reward is [[..]] total reward is [[..]]
            if patient_filter and not line.startswith(patient_filter):
                continue
            step_match = re.search(r'the\s+(\d+)th\s+training', line)
            if not step_match:
                continue
            step = int(step_match.group(1))

            def extract_first_float(tag):
                m = re.search(tag + r':\[\[(.*?)\]\]', line)
                if not m:
                    return np.nan
                try:
                    vals = [float(x) for x in m.group(1).replace(',', ' ').split()]
                    return float(vals[0]) if vals else np.nan
                except Exception:
                    return np.nan

            # V18 是向量，这里取均值
            v18_str = re.search(r'V18:\[\[(.*?)\]\]', line)
            v18_val = np.nan
            if v18_str:
                try:
                    arr = [float(x) for x in v18_str.group(1).replace(',', ' ').split()]
                    if arr:
                        v18_val = float(np.mean(arr))
                except Exception:
                    v18_val = np.nan

            d95_val = extract_first_float('D95')
            dose_t = extract_first_float('dose_target')
            dose_o = extract_first_float('dose_oar_mean')

            steps.append(step)
            d95.append(d95_val)
            v18_mean.append(v18_val)
            dose_ptv.append(dose_t)
            dose_oar_mean.append(dose_o)

    # 排序按step
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    dose_ptv = np.array(dose_ptv)[order]
    dose_oar_mean = np.array(dose_oar_mean)[order]
    d95 = np.array(d95)[order]
    v18_mean = np.array(v18_mean)[order]
    return steps, dose_ptv, dose_oar_mean, d95, v18_mean


def plot_dose_trend(exp_dir: str, episode: int, patient_id: str = None, out_path: str = None):
    csv_path = os.path.join(exp_dir, 'train_info', f'train_{episode}_rounds.csv')
    steps, dose_ptv, dose_oar_mean, d95, v18_mean = parse_train_csv(csv_path, patient_filter=patient_id)
    if len(steps) == 0:
        print(f'No data found in {csv_path} for patient {patient_id or "ALL"}')
        return

    plt.figure(figsize=(8,5))
    # 优先使用记录的处方剂量；若缺失则回退到 D95 作为PTV剂量代理，V18均值做OAR代理（标注单位差异）
    if np.all(np.isnan(dose_ptv)):
        y_ptv = d95
        label_ptv = 'PTV (D95 proxy)'
    else:
        y_ptv = dose_ptv
        label_ptv = 'PTV dose'
    if np.all(np.isnan(dose_oar_mean)):
        y_oar = v18_mean
        label_oar = 'OAR mean (V18 proxy)'
    else:
        y_oar = dose_oar_mean
        label_oar = 'OAR mean dose'

    plt.plot(steps, y_ptv, 'm--', linewidth=2, label=label_ptv)
    plt.plot(steps, y_oar, linewidth=2, label=label_oar)
    plt.xlabel('Step')
    plt.ylabel('Dose (Gy)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:  # 如果out_path包含目录路径
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Dose trend saved to: {out_path}')
        plt.close()
    else:
        plt.show()


def compute_dvh_for_checkpoint(patient_id: str, ckpt_path: str, device: torch.device):
    # 从可视化脚本借用：初始化环境并用模型给出动作，然后执行一步，取t与t+1的DVH
    from visualize_results import process_dvh  # already imported
    env = TP(num_of_organ=1)
    # 推断 action_dim / state_dim 由 ckpt 内部参数
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = state_dict['actor'] if 'actor' in state_dict else state_dict
    # 简单推断（与 visualize_results 中一致）
    mean_w = None
    for k, v in actor_state.items():
        if k.endswith('mean_head.weight'):
            mean_w = v
            break
    if mean_w is None:
        raise RuntimeError('Cannot infer action_dim from checkpoint')
    action_dim = mean_w.shape[0]

    # reset to get initial state and dose
    f_data0, d_range0, dose, V18, D95 = env.reset(patient_id, num_of_organ=action_dim)

    # build actor and load
    from train_portpy import DDPGActor
    actor = DDPGActor(state_dim=action_dim*5, action_dim=action_dim, action_low=0.8, action_high=1.2).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()

    # state vector
    from train_portpy import func_to_state
    state0 = func_to_state(np.array(f_data0), np.array(d_range0)).reshape(-1)
    if len(state0) < action_dim*5:
        state0 = np.pad(state0, (0, action_dim*5 - len(state0)))
    with torch.no_grad():
        s_t = torch.as_tensor(state0, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = actor(s_t, deterministic=True)
        action = action.squeeze(0).cpu().numpy()
        action = action[:action_dim]

    f_data1, d_range1, reward, dose1, done, V18_1, D95_1 = env.step(patient_id, action, dose, V18, D95)
    return (np.array(d_range0), np.array(f_data0), np.array(d_range1), np.array(f_data1))


def infer_action_dim_from_ckpt(ckpt_path: str, device: torch.device) -> int:
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = sd['actor'] if 'actor' in sd else sd
    for k, v in actor_state.items():
        if k.endswith('mean_head.weight'):
            return int(v.shape[0])
    raise RuntimeError('Cannot infer action_dim from checkpoint')


def infer_action_range_from_ckpt(ckpt_path: str, device: torch.device):
    """从checkpoint中读取action范围，如果不存在则使用默认值"""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = sd['actor'] if 'actor' in sd else sd
    
    # PyTorch的register_buffer会在state_dict中保存，键名就是buffer的名字
    action_low = None
    action_high = None
    
    # 检查actor_state中的所有键
    for key in actor_state.keys():
        if 'action_low' in key.lower():
            val = actor_state[key]
            if isinstance(val, torch.Tensor):
                action_low = float(val.item())
            else:
                action_low = float(val)
        elif 'action_high' in key.lower():
            val = actor_state[key]
            if isinstance(val, torch.Tensor):
                action_high = float(val.item())
            else:
                action_high = float(val)
    
    # 如果找到了，返回
    if action_low is not None and action_high is not None:
        return action_low, action_high
    
    # 如果都没有，使用默认值（与训练时一致）
    print(f"[警告] 无法从checkpoint读取action范围，使用默认值 [0.8, 1.2]")
    return 0.8, 1.2


def get_local_indices_for_ptv_and_oar(patient_id: str):
    from portpy_functions import load_patient_data
    cst, ct, inf_matrix, pln = load_patient_data(patient_id, opt_vox_res_mm=None)
    struct_names = list(cst.structures.keys())
    cst_Inx = cst.get_constraint_indices()
    ptv_local = None
    oar_local = []
    for local_idx, global_idx in enumerate(cst_Inx):
        name = struct_names[global_idx].upper()
        if 'PTV' in name and ptv_local is None:
            ptv_local = local_idx
        elif ('GTV' in name) or ('CTV' in name):
            continue
        else:
            oar_local.append(local_idx)
    return ptv_local, oar_local


def rollout_dose_trend_with_ckpt(patient_id: str, ckpt_path: str, steps: int, out_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_dim = infer_action_dim_from_ckpt(ckpt_path, device)
    action_low, action_high = infer_action_range_from_ckpt(ckpt_path, device)
    print(f"[Rollout] 从checkpoint读取: action_dim={action_dim}, action_range=[{action_low}, {action_high}]")
    
    ptv_local, oar_local = get_local_indices_for_ptv_and_oar(patient_id)
    if ptv_local is None:
        print('Warning: No PTV found; dose_target will be NaN')
    if not oar_local:
        print('Warning: No OAR indices; dose_oar_mean will be NaN')

    env = TP(num_of_organ=action_dim)
    f_data0, d_range0, dose, V18, D95 = env.reset(patient_id, num_of_organ=action_dim)
    from train_portpy import DDPGActor, func_to_state
    actor = DDPGActor(state_dim=action_dim*5, action_dim=action_dim, action_low=action_low, action_high=action_high).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = state_dict['actor'] if 'actor' in state_dict else state_dict
    actor.load_state_dict(actor_state)
    actor.eval()

    step_ids = []
    dose_ptv = []
    dose_oar_mean = []
    
    # 初始状态调试
    print(f"[Rollout调试] 初始状态: f_data0形状={np.array(f_data0).shape}, d_range0形状={np.array(d_range0).shape}")
    print(f"[Rollout调试] 初始剂量: dose形状={np.array(dose).shape if hasattr(dose, '__len__') else 'scalar'}")
    
    for t in range(1, steps+1):
        # 确保f_data0是2D数组
        f_data0_arr = np.array(f_data0)
        d_range0_arr = np.array(d_range0).flatten()
        
        if f_data0_arr.ndim == 1:
            # 如果是1D，需要reshape
            print(f"[Rollout警告] Step {t}: f_data0是1D数组，形状={f_data0_arr.shape}，尝试reshape")
            # 假设每个器官有相同数量的点
            num_points = len(f_data0_arr) // action_dim
            if len(f_data0_arr) % action_dim == 0:
                f_data0_arr = f_data0_arr.reshape(action_dim, num_points)
            else:
                raise ValueError(f"无法将f_data0 reshape为2D数组: {f_data0_arr.shape}, action_dim={action_dim}")
        
        state_mat = func_to_state(f_data0_arr, d_range0_arr)
        state_vec = state_mat.reshape(-1)
        
        if len(state_vec) < action_dim*5:
            print(f"[Rollout调试] Step {t}: 状态维度不足 ({len(state_vec)} < {action_dim*5})，用0填充")
            state_vec = np.pad(state_vec, (0, action_dim*5 - len(state_vec)))
        elif len(state_vec) > action_dim*5:
            print(f"[Rollout警告] Step {t}: 状态维度超出 ({len(state_vec)} > {action_dim*5})，截断")
            state_vec = state_vec[:action_dim*5]
        
        with torch.no_grad():
            s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = actor(s_t, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
            action = action[:action_dim]
        
        print(f"[Rollout调试] Step {t}: action范围=[{action.min():.3f}, {action.max():.3f}], mean={action.mean():.3f}")
        print(f"[Rollout调试] Step {t}: 前3个action值={action[:3]}")
        print(f"[Rollout调试] Step {t}: 当前剂量范围=[{dose.min():.3f}, {dose.max():.3f}]" if hasattr(dose, '__len__') else f"[Rollout调试] Step {t}: 当前剂量={dose}")
        
        f_data1, d_range1, reward, dose1, done, V18_1, D95_1 = env.step(patient_id, action, dose, V18, D95)
        
        print(f"[Rollout调试] Step {t}: 新剂量范围=[{dose1.min():.3f}, {dose1.max():.3f}]" if hasattr(dose1, '__len__') else f"[Rollout调试] Step {t}: 新剂量={dose1}")

        if hasattr(dose1, '__len__'):
            ptv_val = float(dose1[ptv_local]) if ptv_local is not None and ptv_local < len(dose1) else np.nan
            oar_vals = [float(dose1[i]) for i in oar_local if i < len(dose1)] if oar_local else []
            oar_mean = float(np.mean(oar_vals)) if oar_vals else np.nan
        else:
            ptv_val = np.nan
            oar_mean = np.nan

        step_ids.append(t)
        dose_ptv.append(ptv_val)
        dose_oar_mean.append(oar_mean)

        f_data0, d_range0, dose, V18, D95 = f_data1, d_range1, dose1, V18_1, D95_1

    plt.figure(figsize=(8,5))
    plt.plot(step_ids, dose_ptv, 'm--', linewidth=2, label='PTV dose')
    plt.plot(step_ids, dose_oar_mean, linewidth=2, label='OAR mean dose')
    plt.xlabel('Step')
    plt.ylabel('Dose (Gy)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_path:
        # 确保输出目录存在
        out_dir = os.path.dirname(out_path)
        if out_dir:  # 如果out_path包含目录路径
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Dose trend (rollout) saved to: {out_path}')
        plt.close()
    else:
        plt.show()


def plot_dvh_snapshots(exp_dir: str, episode: int, patient_id: str, steps: list, out_path: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始: 使用该episode下第一个step的快照作为初始化近似（或重新跑reset的t=0）
    ckpts = []
    for s in steps:
        # actor_ep{episode}_step{s}.pth
        candidate = os.path.join(exp_dir, 'actor', f'actor_ep{episode}_step{s}.pth')
        if os.path.exists(candidate):
            ckpts.append(candidate)
        else:
            print(f'Warning: checkpoint not found: {candidate}')

    if not ckpts:
        print('No checkpoints found for snapshots.')
        return

    # 先获取t=0曲线
    env = TP(num_of_organ=1)
    # 使用第一个ckpt推断action_dim
    sd = torch.load(ckpts[0], map_location=device, weights_only=False)
    actor_state = sd['actor'] if 'actor' in sd else sd
    mean_w = None
    for k, v in actor_state.items():
        if k.endswith('mean_head.weight'):
            mean_w = v
            break
    action_dim = mean_w.shape[0]
    f_data0, d_range0, dose, V18, D95 = env.reset(patient_id, num_of_organ=action_dim)

    # 处理target与OAR mean
    target_idx = 0
    oar_indices = list(range(1, action_dim))
    d0_t, v0_t = process_dvh(d_range0, f_data0[target_idx])
    # OAR mean
    def mean_oar(d_range, f_data):
        processed = []
        max_d = 0.0
        for idx in oar_indices:
            d_r, v = process_dvh(d_range, f_data[idx])
            processed.append((d_r, v))
            if len(d_r) > 0:
                max_d = max(max_d, float(np.max(d_r)))
        if max_d <= 0:
            return np.array([]), np.array([])
        grid = np.linspace(0.0, max_d, 200)
        vals = []
        for d_r, v in processed:
            if len(d_r) == 0:
                continue
            vv = np.interp(grid, d_r, v)
            for j in range(1, len(vv)):
                if vv[j] > vv[j-1]:
                    vv[j] = vv[j-1]
            vals.append(vv)
        if not vals:
            return np.array([]), np.array([])
        return grid, np.mean(np.stack(vals, axis=0), axis=0)

    d0_o, v0_o = mean_oar(d_range0, f_data0)

    plt.figure(figsize=(12,4))
    # Target subplot
    plt.subplot(1,2,1)
    plt.plot(d0_t, v0_t, label='Target init', linewidth=2)
    colors = plt.cm.viridis(np.linspace(0,1,max(1,len(ckpts))))
    for color, ck in zip(colors, ckpts):
        d0, f0, d1, f1 = compute_dvh_for_checkpoint(patient_id, ck, device)
        dt, vt = process_dvh(d1, f1[target_idx])
        step_match = re.search(r'_step(\d+)\.pth$', ck)
        step_name = step_match.group(1) if step_match else '?'
        plt.plot(dt, vt, color=color, linestyle='--', label=f'Target step{step_name}', linewidth=2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('Target DVH')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # OAR mean subplot
    plt.subplot(1,2,2)
    plt.plot(d0_o, v0_o, label='OAR mean init', linewidth=2)
    for color, ck in zip(colors, ckpts):
        d0, f0, d1, f1 = compute_dvh_for_checkpoint(patient_id, ck, device)
        do, vo = mean_oar(d1, f1)
        step_match = re.search(r'_step(\d+)\.pth$', ck)
        step_name = step_match.group(1) if step_match else '?'
        plt.plot(do, vo, color=color, linestyle='--', label=f'OAR mean step{step_name}', linewidth=2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('OAR Mean DVH')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:  # 如果out_path包含目录路径
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'DVH snapshots saved to: {out_path}')
        plt.close()
    else:
        plt.show()


def rollout_dvh_snapshots(patient_id: str, ckpt_path: str, steps: list, out_path: str = None):
    """
    使用同一个checkpoint在同一病人上连续rollout，绘制 t=0 以及指定步数的DVH（Target=PTV，OAR=mean OAR）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_dim = infer_action_dim_from_ckpt(ckpt_path, device)

    # 结构索引（在 cst_Inx 的局部索引空间）：PTV 单独，OAR 排除 GTV/CTV
    ptv_local, oar_local = get_local_indices_for_ptv_and_oar(patient_id)
    if ptv_local is None:
        print('Warning: No PTV found, DVH target曲线将跳过。')

    # 环境与actor
    env = TP(num_of_organ=action_dim)
    f_data, d_range, dose, V18, D95 = env.reset(patient_id, num_of_organ=action_dim)
    from train_portpy import DDPGActor, func_to_state
    actor = DDPGActor(state_dim=action_dim*5, action_dim=action_dim, action_low=0.8, action_high=1.2).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = sd['actor'] if 'actor' in sd else sd
    actor.load_state_dict(actor_state)
    actor.eval()

    # 工具：处理Target与OAR mean曲线
    def mean_oar(d_range_arr, f_data_arr):
        if not oar_local:
            return np.array([]), np.array([])
        processed = []
        max_d = 0.0
        for idx in oar_local:
            if idx >= len(f_data_arr):
                continue
            d_r, v = process_dvh(d_range_arr, f_data_arr[idx])
            processed.append((d_r, v))
            if len(d_r) > 0:
                max_d = max(max_d, float(np.max(d_r)))
        if max_d <= 0 or not processed:
            return np.array([]), np.array([])
        grid = np.linspace(0.0, max_d, 200)
        vals = []
        for d_r, v in processed:
            if len(d_r) == 0:
                continue
            vv = np.interp(grid, d_r, v)
            for j in range(1, len(vv)):
                if vv[j] > vv[j-1]:
                    vv[j] = vv[j-1]
            vals.append(vv)
        if not vals:
            return np.array([]), np.array([])
        return grid, np.mean(np.stack(vals, axis=0), axis=0)

    # 收集快照：t=0 + 指定步
    desired = sorted(set([s for s in steps if isinstance(s, int) and s >= 0]))
    snapshots = {}
    # t=0
    if ptv_local is not None and ptv_local < len(f_data):
        d_t0, v_t0 = process_dvh(d_range, f_data[ptv_local])
    else:
        d_t0, v_t0 = np.array([]), np.array([])
    d_o0, v_o0 = mean_oar(d_range, f_data)
    snapshots[0] = (d_t0, v_t0, d_o0, v_o0)

    max_step = max(desired) if desired else 0
    for t in range(1, max_step + 1):
        # 计算动作并前进一步
        state_vec = func_to_state(np.array(f_data), np.array(d_range)).reshape(-1)
        if len(state_vec) < action_dim*5:
            state_vec = np.pad(state_vec, (0, action_dim*5 - len(state_vec)))
        with torch.no_grad():
            s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = actor(s_t, deterministic=True)
            action = action.squeeze(0).cpu().numpy()
            action = action[:action_dim]
        f_data, d_range, reward, dose, done, V18, D95 = env.step(patient_id, action, dose, V18, D95)

        if t in desired:
            if ptv_local is not None and ptv_local < len(f_data):
                d_t, v_t = process_dvh(d_range, f_data[ptv_local])
            else:
                d_t, v_t = np.array([]), np.array([])
            d_o, v_o = mean_oar(d_range, f_data)
            snapshots[t] = (d_t, v_t, d_o, v_o)

    # 画图
    plt.figure(figsize=(12, 4))
    # Target
    plt.subplot(1, 2, 1)
    if len(snapshots[0][0]) > 0:
        plt.plot(snapshots[0][0], snapshots[0][1], label='Target t=0', linewidth=2)
    colors = plt.cm.plasma(np.linspace(0, 1, len(desired)))
    for c, s in zip(colors, desired):
        d_t, v_t, _, _ = snapshots.get(s, (np.array([]), np.array([]), np.array([]), np.array([])))
        if len(d_t) > 0:
            plt.plot(d_t, v_t, linestyle='--', color=c, linewidth=2, label=f'Target t={s}')
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('Target (PTV) DVH')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # OAR mean
    plt.subplot(1, 2, 2)
    if len(snapshots[0][2]) > 0:
        plt.plot(snapshots[0][2], snapshots[0][3], label='OAR mean t=0', linewidth=2)
    for c, s in zip(colors, desired):
        _, _, d_o, v_o = snapshots.get(s, (np.array([]), np.array([]), np.array([]), np.array([])))
        if len(d_o) > 0:
            plt.plot(d_o, v_o, linestyle='--', color=c, linewidth=2, label=f'OAR mean t={s}')
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('OAR Mean DVH')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:  # 如果out_path包含目录路径
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'DVH rollout snapshots saved to: {out_path}')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', default=None, help='Experiment ID, e.g., exp_005')
    parser.add_argument('--episode', type=int, default=1, help='Episode number to read CSV/checkpoints')
    parser.add_argument('--patient', default=None, help='Patient ID to filter, e.g., Lung_Patient_2')
    parser.add_argument('--trend-out', default=None, help='Path to save dose trend figure')
    parser.add_argument('--dvh-steps', default='10,20,30', help='Comma-separated steps for DVH snapshots')
    parser.add_argument('--dvh-out', default=None, help='Path to save DVH snapshots figure')
    parser.add_argument('--rollout-ckpt', default=None, help='If set, roll out this checkpoint for N steps to draw trend and/or DVH')
    parser.add_argument('--rollout-steps', type=int, default=30, help='Steps for rollout trend')
    args = parser.parse_args()

    exp_dir = os.path.join(os.path.dirname(__file__), 'result', args.exp_id) if args.exp_id else None

    # Trend: prefer rollout mode with checkpoint (post-training evaluation)
    if args.rollout_ckpt and args.patient:
        rollout_dose_trend_with_ckpt(args.patient, args.rollout_ckpt, args.rollout_steps, out_path=args.trend_out)
    elif exp_dir:
        plot_dose_trend(exp_dir, args.episode, patient_id=args.patient, out_path=args.trend_out)
    else:
        print('Skip trend: provide --rollout-ckpt with --patient, or --exp-id to read CSV')

    # DVH snapshots: prefer rollout with single final model when ckpt provided
    steps = [int(s) for s in args.dvh_steps.split(',') if s.strip().isdigit()]
    if args.rollout_ckpt and args.patient:
        rollout_dvh_snapshots(args.patient, args.rollout_ckpt, steps, out_path=args.dvh_out)
    elif exp_dir and args.patient:
        plot_dvh_snapshots(exp_dir, args.episode, args.patient, steps, out_path=args.dvh_out)
    else:
        print('Skip DVH snapshots: please specify --patient and either --rollout-ckpt or --exp-id')


if __name__ == '__main__':
    main()


