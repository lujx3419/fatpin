#!/usr/bin/env python3
"""
可视化训练结果脚本

功能：
1. 绘制奖励曲线（reward curve）：从训练日志CSV文件中提取奖励值并可视化
2. DVH对比图：加载指定的模型checkpoint，对比某病人在时间步 t 和 t+1 的DVH曲线
   
   DVH对比说明：
   - t时刻：通过env.reset()获得的患者初始DVH曲线（当前状态）
   - t+1时刻：模型根据当前状态预测动作，执行env.step()后得到的DVH曲线（执行一步后的状态）
   - 对比这两条曲线可以直观看到模型在当前状态下执行一步动作的效果
   - 注意：目前只支持对比相邻的两个时间步（t和t+1），因为需要模型预测动作并执行

使用示例：

# ========== 场景1: 只查看奖励曲线（最简单）==========
# 在屏幕上显示奖励曲线
python3 visualize_results.py --exp-id exp_001 --only-reward

# 将奖励曲线保存到文件（自动保存到 result/exp_001/visualization/）
# 注意：如果不指定--output-dir，默认保存到 result/exp_001/visualization/
# 如果指定相对路径，会相对于 result/exp_001/ 目录

python3 visualize_results.py --exp-id exp_001 --only-reward --output-dir visualization

# 既显示又保存奖励曲线
python3 visualize_results.py --exp-id exp_001 --only-reward --output-dir visualization --show


# ========== 场景2: 查看DVH对比（需要checkpoint）==========
# DVH对比展示的是：模型在当前状态下执行一步动作前后的DVH变化（t vs t+1）
# 
# 重要说明：
# - checkpoint是通用模型，可以在任何患者上测试，不绑定特定患者
# - --patient参数指定要在哪个患者上测试/可视化模型的效果
# - 可以在同一checkpoint上测试不同的患者，查看模型在不同患者上的表现

# 方式1: 使用单独保存的actor checkpoint，不指定输出目录（默认保存到result/exp_001/visualization/）
python3 visualize_results.py --exp-id exp_001 --patient Lung_Patient_2 --ckpt actor/actor_ep30_step1.pth

# 方式2: 指定相对路径的输出目录（会自动保存到result/exp_001/visualization/）
python3 visualize_results.py --exp-id exp_001 --patient Lung_Patient_2 --ckpt actor/actor_ep30_step1.pth --output-dir visualization

# 方式3: 在同一checkpoint上测试不同患者（展示模型在不同患者上的表现）
python3 visualize_results.py --exp-id exp_001 --patient Lung_Patient_3 --ckpt actor/actor_ep30_step1.pth --output-dir visualization

# 方式4: 使用捆绑保存的完整checkpoint（包含actor、critic等所有模型状态）
# 脚本会自动识别格式并从字典中提取actor
python3 visualize_results.py --exp-id exp_001 --patient Lung_Patient_2 --ckpt actor_critic_ddpg_30_1.pth --output-dir visualization

# 两种checkpoint格式的区别：
# - 单独保存: actor/actor_ep30_step1.pth（只包含actor的权重，文件小，加载快）
# - 捆绑保存: actor_critic_ddpg_30_1.pth（包含完整模型状态，可用于恢复训练）
# 对于DVH可视化，两种格式都可以使用，脚本会自动识别并加载actor部分


# ========== 场景3: 使用绝对路径指定checkpoint和输出目录 ==========
python3 visualize_results.py --exp-id exp_001 --patient Lung_Patient_2 \
    --ckpt /full/path/to/result/exp_001/actor/actor_ep30_step1.pth \
    --output-dir /full/path/to/result/exp_001/visualization


# ========== 场景4: 使用旧格式（不使用--exp-id）==========
# 适用于旧版本的训练脚本生成的结果
python3 visualize_results.py --csv-dir result/train_info --patient Lung_Patient_2 \
    --ckpt result/actor/multi_30_1.pth \
    --output-dir result/visualization


参数说明：
  --exp-id:        实验ID，对应 result/{exp_id}/ 目录
  --csv-dir:       手动指定CSV日志目录（如果不使用--exp-id）
  --patient:       患者ID，指定要在哪个患者上测试/可视化模型（默认 'Lung_Patient_2'）
                   注意：checkpoint是通用模型，不绑定特定患者，可以在不同患者上测试
                   可用值：'Lung_Patient_2', 'Lung_Patient_3', 'Lung_Patient_4', 'Lung_Patient_5'
  --ckpt:          Checkpoint路径，支持两种格式：
                    1. 单独保存: actor/actor_ep30_step1.pth（只包含actor权重）
                    2. 捆绑保存: actor_critic_ddpg_30_1.pth（包含完整模型状态）
                   可以是相对路径（相对于result/{exp_id}/）或绝对路径
                   脚本会自动识别格式并从捆绑格式中提取actor
                   注意：checkpoint文件名中的episode和step是训练时的轮次和步数，
                   不代表该checkpoint绑定特定患者，可以在任何患者上使用
  --only-reward:   只绘制奖励曲线，不绘制DVH对比
  --output-dir:    保存图片的目录（如果不指定，默认只在屏幕上显示）
  --show:          在保存的同时也在屏幕上显示图片

注意：
- DVH对比展示的是"当前状态(t)"和"执行一步动作后(t+1)"的DVH曲线变化
- 目前只支持对比相邻的两个时间步（t和t+1），因为需要模型预测并执行动作
- 如果使用 --exp-id，checkpoint路径可以使用相对路径（如 actor/actor_ep30_step1.pth）
- 脚本会自动在 result/{exp_id}/ 目录下查找checkpoint和训练日志
- **输出路径规则**：
  * 如果不指定 --output-dir：默认保存到 result/{exp_id}/visualization/（不显示）
  * 如果指定相对路径（如 visualization 或 visualization/subfolder）：相对于 result/{exp_id}/ 目录
  * 如果指定绝对路径：直接使用该路径
- 如果不指定 --output-dir，图片只会在屏幕上显示
- 如果指定了 --output-dir，默认只保存不显示，需要加上 --show 才能同时显示

"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(__file__))
from train_portpy import TP, DDPGActor, func_to_state, device


def process_dvh(d_range, volume_points):
    """
    处理DVH数据，确保：
    1. 数据按剂量排序（从小到大）
    2. 从100%开始（在0剂量时）
    3. 单调递减（随剂量增加，体积减少）
    4. 插值处理，使曲线更平滑（避免快速下降）
    """
    # 确保是numpy数组
    d_range = np.array(d_range).flatten()
    volume_points = np.array(volume_points).flatten()
    
    # 按剂量排序
    sort_idx = np.argsort(d_range)
    d_range_sorted = d_range[sort_idx]
    volume_sorted = volume_points[sort_idx]
    
    # 确保从100%开始（如果第一个点不是0剂量或不是100%）
    if len(d_range_sorted) > 0:
        if d_range_sorted[0] > 0.01:  # 如果第一个点不是0剂量
            # 在0剂量处插入100%
            d_range_sorted = np.concatenate([[0.0], d_range_sorted])
            volume_sorted = np.concatenate([[100.0], volume_sorted])
        elif volume_sorted[0] < 99.0:  # 如果第一个点是0剂量但不是100%
            volume_sorted[0] = 100.0
    
    # 确保单调递减（累积DVH应该是递减的）
    for i in range(1, len(volume_sorted)):
        if volume_sorted[i] > volume_sorted[i-1]:
            volume_sorted[i] = volume_sorted[i-1]
    
    # 确保数据范围在[0, 100]
    volume_sorted = np.clip(volume_sorted, 0, 100)
    
    # 诊断信息（帮助理解快速下降的原因）
    if len(d_range_sorted) < 50:
        print(f"警告: DVH数据点较少 ({len(d_range_sorted)}个点)，可能导致曲线快速下降")
        print(f"  剂量范围: {d_range_sorted[0]:.3f} - {d_range_sorted[-1]:.3f} Gy")
        print(f"  前5个点的剂量: {d_range_sorted[:5]}")
        print(f"  前5个点的体积: {volume_sorted[:5]}")
        
        # 创建更密集的剂量网格进行插值
        max_dose = np.max(d_range_sorted)
        dose_grid_interp = np.linspace(0, max_dose, max(100, len(d_range_sorted) * 2))
        
        # 使用线性插值（保持单调性）
        volume_interp = np.interp(dose_grid_interp, d_range_sorted, volume_sorted)
        
        # 确保插值后仍然单调递减
        for i in range(1, len(volume_interp)):
            if volume_interp[i] > volume_interp[i-1]:
                volume_interp[i] = volume_interp[i-1]
        
        print(f"  插值后点数: {len(dose_grid_interp)}")
        d_range_sorted = dose_grid_interp
        volume_sorted = volume_interp
    else:
        # 即使数据点多，也检查是否有快速下降
        # 检查前10%的剂量范围内体积下降了多少
        if len(d_range_sorted) > 10:
            idx_10pct = int(len(d_range_sorted) * 0.1)
            volume_drop = volume_sorted[0] - volume_sorted[idx_10pct]
            dose_range = d_range_sorted[idx_10pct] - d_range_sorted[0]
            if dose_range > 0:
                drop_rate = volume_drop / dose_range
                if drop_rate > 50:  # 每Gy下降超过50%
                    print(f"警告: DVH曲线在低剂量区域下降很快")
                    print(f"  前10%剂量范围 ({d_range_sorted[0]:.3f} - {d_range_sorted[idx_10pct]:.3f} Gy)")
                    print(f"  体积从 {volume_sorted[0]:.1f}% 降到 {volume_sorted[idx_10pct]:.1f}%")
                    print(f"  下降速率: {drop_rate:.1f} %/Gy (通常应该在10-30 %/Gy)")
                    print(f"  这可能意味着:")
                    print(f"    1. 剂量分布非常集中（大部分体积在很小剂量范围）")
                    print(f"    2. DVH数据点分布不均匀（低剂量区域点少）")
                    print(f"    3. 剂量计算可能有问题")
    
    return d_range_sorted, volume_sorted


def plot_rewards(csv_dir: str, output_dir: str = None, show: bool = True) -> None:
    files = sorted(glob.glob(os.path.join(csv_dir, 'train_*_rounds.csv')))
    steps, rewards = [], []
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.search(r'reward is\s+(\[?\[?([\-\d.eE]+))', line)
                if m:
                    rewards.append(float(m.group(2)))
                    steps.append(len(rewards))
    if not rewards:
        print('No rewards found in CSV. Skipping reward plot.')
        return
    plt.figure(figsize=(6,4))
    plt.plot(steps, rewards, label='reward per step')
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'reward_curve.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Reward curve saved to: {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dvh(patient_id: str, ckpt_path: str, output_dir: str = None, show: bool = True) -> None:
    """
    绘制DVH对比图：展示模型在给定状态下执行一步动作前后的DVH变化
    
    对比的是：
    - t时刻（初始状态）：通过env.reset()获得的初始DVH
    - t+1时刻（执行动作后）：模型预测动作，执行env.step()后得到的DVH
    
    这样可以直观地看到模型在当前状态下执行一步动作的效果。
    
    Args:
        patient_id: 患者ID，指定要在哪个患者上测试模型（例如：'Lung_Patient_2'）
                   注意：checkpoint是通用模型，可以在任何患者上测试
        ckpt_path: Checkpoint路径，支持两种格式：
                  1. 单独的actor checkpoint: actor/actor_ep30_step1.pth
                  2. 捆绑的checkpoint: actor_critic_ddpg_30_1.pth（包含完整的模型状态）
                  checkpoint不绑定特定患者，可以在不同患者上使用
        output_dir: 输出目录
        show: 是否显示图片
    """
    # 先加载checkpoint以推断维度
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 获取actor的state_dict
    if isinstance(ckpt_data, dict) and 'actor' in ckpt_data:
        actor_state_dict = ckpt_data['actor']
        print(f"Loaded actor from bundled checkpoint (episode: {ckpt_data.get('episode', 'unknown')}, step: {ckpt_data.get('step', 'unknown')})")
    else:
        actor_state_dict = ckpt_data
        print(f"Loaded actor from standalone checkpoint")
    
    # 从checkpoint推断action_dim（通过mean_head.weight的形状）
    if 'mean_head.weight' in actor_state_dict:
        # mean_head.weight形状是 [action_dim, 256]
        action_dim = actor_state_dict['mean_head.weight'].shape[0]
        print(f"[推断] 从checkpoint推断 action_dim = {action_dim}")
    elif 'fc1.weight' in actor_state_dict:
        # fc1.weight形状是 [256, action_dim]（在旧版本中可能是这样）
        action_dim = actor_state_dict['fc1.weight'].shape[1]
        print(f"[推断] 从checkpoint推断 action_dim = {action_dim} (通过fc1.weight)")
    else:
        # 如果无法推断，尝试使用患者数据的实际结构数量
        print("[警告] 无法从checkpoint推断action_dim，尝试从患者数据推断...")
        try:
            from portpy_structures import load_patient_data
            cst, _, _, _ = load_patient_data(patient_id, opt_vox_res_mm=None)
            cst_Inx = cst.get_constraint_indices()
            action_dim = len(cst_Inx)
            print(f"[推断] 从患者数据推断 action_dim = {action_dim} (有约束的结构数量)")
        except Exception as e:
            print(f"[错误] 无法推断action_dim: {e}")
            # 默认值（旧代码的维度）
            action_dim = 9
            print(f"[回退] 使用默认值 action_dim = {action_dim}")
    
    # 推断state_dim（通过organ_embed.weight的形状）
    per_organ_state_dim = 5  # 默认值
    if 'organ_embed.weight' in actor_state_dict:
        # organ_embed.weight形状是 [1, per_organ_k]
        # per_organ_k = state_dim // action_dim
        per_organ_k = actor_state_dict['organ_embed.weight'].shape[1]
        state_dim = per_organ_k * action_dim
        per_organ_state_dim = per_organ_k
        print(f"[推断] 从organ_embed推断 per_organ_k = {per_organ_k}, state_dim = {state_dim}")
    elif 'fc1.weight' in actor_state_dict:
        # fc1的输入维度是action_dim（因为输入是h(M)，即num_organs维）
        # 但fc1的输入是h，不是state，所以无法直接推断state_dim
        # 使用默认的per_organ_state_dim
        state_dim = action_dim * per_organ_state_dim
        print(f"[推断] 使用默认 per_organ_state_dim = {per_organ_state_dim}")
    else:
        state_dim = action_dim * per_organ_state_dim
        print(f"[推断] 使用默认 per_organ_state_dim = {per_organ_state_dim}")
    
    print(f"[推断] state_dim = {state_dim}, action_dim = {action_dim}, per_organ_state_dim = {per_organ_state_dim}")
    
    action_low, action_high = 0.95, 1.05
    
    # 使用推断的维度创建actor
    actor = DDPGActor(state_dim, action_dim, action_low, action_high).to(device)
    
    # 加载checkpoint
    actor.load_state_dict(actor_state_dict)
    
    actor.eval()

    # 使用推断的action_dim作为器官数量（对应有约束的结构数量）
    env = TP(num_of_organ=action_dim)
    f_data0, d_range0, dose, V18, D95 = env.reset(patient_id, num_of_organ=action_dim)
    state0 = func_to_state(np.array(f_data0), np.array(d_range0)).reshape(-1)

    with torch.no_grad():
        s_t = torch.as_tensor(state0, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = actor(s_t, deterministic=True)
        action = action.squeeze(0).cpu().numpy()

    f_data1, d_range1, reward, dose1, done, V18_1, D95_1 = env.step(patient_id, action, dose, V18, D95)

    target_idx = 0
    # OAR索引：排除target（索引0），包括剩余的结构
    oar_indices = list(range(1, action_dim))

    # 处理Target DVH数据
    d_range0_target, f_data0_target = process_dvh(d_range0, f_data0[target_idx])
    d_range1_target, f_data1_target = process_dvh(d_range1, f_data1[target_idx])
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(d_range0_target, f_data0_target, label='Target t', linewidth=2)
    plt.plot(d_range1_target, f_data1_target, label='Target t+1', linewidth=2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('Target DVH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max(np.max(d_range0_target), np.max(d_range1_target))])

    # 处理OAR DVH数据（先对每个OAR做处理，再在统一网格上取均值，避免直接逐点平均造成“直线”）
    def mean_oar_dvh(d_range, f_data, indices):
        # 先对每个OAR做排序/单调化处理
        processed = []
        max_dose = 0.0
        for idx in indices:
            d_r, v = process_dvh(d_range, f_data[idx])
            processed.append((d_r, v))
            if len(d_r) > 0:
                max_dose = max(max_dose, float(np.max(d_r)))
        if max_dose <= 0:
            return np.array([]), np.array([])
        # 统一到公共网格
        common_grid = np.linspace(0.0, max_dose, 200)
        vals = []
        for d_r, v in processed:
            if len(d_r) == 0:
                continue
            vv = np.interp(common_grid, d_r, v)
            # 再次保证单调递减
            for j in range(1, len(vv)):
                if vv[j] > vv[j-1]:
                    vv[j] = vv[j-1]
            vals.append(vv)
        if not vals:
            return np.array([]), np.array([])
        mean_v = np.mean(np.stack(vals, axis=0), axis=0)
        return common_grid, mean_v

    d_range0_oar, oar_mean0_processed = mean_oar_dvh(d_range0, f_data0, oar_indices)
    d_range1_oar, oar_mean1_processed = mean_oar_dvh(d_range1, f_data1, oar_indices)
    
    plt.subplot(1,2,2)
    plt.plot(d_range0_oar, oar_mean0_processed, label='OAR mean t', linewidth=2)
    plt.plot(d_range1_oar, oar_mean1_processed, label='OAR mean t+1', linewidth=2)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title('OAR Mean DVH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, max(np.max(d_range0_oar), np.max(d_range1_oar))])

    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 从checkpoint路径提取episode信息
        ckpt_basename = os.path.basename(ckpt_path)
        # 例如: actor_ep30_step1.pth -> ep30
        ep_match = re.search(r'ep(\d+)', ckpt_basename)
        ep_suffix = f'_ep{ep_match.group(1)}' if ep_match else ''
        patient_safe = patient_id.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'dvh_comparison_{patient_safe}{ep_suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'DVH comparison saved to: {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', default=None, help='Experiment ID. If not specified, look for result/train_info (old format)')
    parser.add_argument('--csv-dir', default=None, help='Directory containing train_*_rounds.csv files. Auto-detected if --exp-id is specified.')
    parser.add_argument('--patient', default='Lung_Patient_2')
    parser.add_argument('--ckpt', default='', help='Checkpoint path. If --exp-id is specified, can use relative path like "actor/actor_ep30_step1.pth"')
    parser.add_argument('--only-reward', action='store_true')
    parser.add_argument('--output-dir', default='', help='Directory to save plots. If not specified, only display.')
    parser.add_argument('--show', action='store_true', help='Display plots (in addition to saving if --output-dir is set)')
    args = parser.parse_args()

    # 确定CSV目录和checkpoint路径
    if args.exp_id:
        # 使用实验目录结构
        exp_base_dir = os.path.join(os.path.dirname(__file__), 'result', args.exp_id)
        csv_dir = args.csv_dir or os.path.join(exp_base_dir, 'train_info')
        
        # 如果checkpoint是相对路径，补充完整路径
        if args.ckpt:
            if os.path.isabs(args.ckpt):
                # 绝对路径，直接使用
                if not os.path.exists(args.ckpt):
                    print(f"Error: Checkpoint file not found: {args.ckpt}")
                    sys.exit(1)
                # 确保使用绝对路径
                args.ckpt = os.path.abspath(args.ckpt)
            else:
                # 相对路径，优先在exp_base_dir下查找
                ckpt_path = os.path.join(exp_base_dir, args.ckpt)
                if os.path.exists(ckpt_path):
                    args.ckpt = os.path.abspath(ckpt_path)
                elif os.path.exists(args.ckpt):
                    # 如果当前目录下存在，使用绝对路径
                    args.ckpt = os.path.abspath(args.ckpt)
                else:
                    print(f"Error: Checkpoint file not found: {args.ckpt}")
                    print(f"  Tried: {os.path.abspath(os.path.join(os.getcwd(), args.ckpt))}")
                    print(f"  Tried: {ckpt_path}")
                    print(f"  Please check that the checkpoint file exists.")
                    print(f"  Exp base directory: {exp_base_dir}")
                    sys.exit(1)
        
        # 如果output_dir为空，默认保存到实验的visualization目录
        if not args.output_dir:
            output_dir = os.path.join(exp_base_dir, 'visualization')
        else:
            # 如果指定了output_dir，检查是否是相对路径
            if os.path.isabs(args.output_dir):
                # 绝对路径，直接使用
                output_dir = args.output_dir
            else:
                # 相对路径，相对于exp_base_dir（确保保存在实验目录内）
                output_dir = os.path.join(exp_base_dir, args.output_dir)
    else:
        # 旧格式：直接使用result目录
        csv_dir = args.csv_dir or os.path.join(os.path.dirname(__file__), 'result', 'train_info')
        output_dir = args.output_dir if args.output_dir else None

    # 如果没有指定output_dir，默认显示；如果指定了output_dir但没有--show，就不显示
    show_plots = args.show or not output_dir
    
    plot_rewards(csv_dir, output_dir=output_dir if output_dir else None, show=show_plots)
    if args.only_reward:
        return
    if not args.ckpt:
        print('No checkpoint specified. Use --ckpt to plot DVH comparison.')
        return
    plot_dvh(args.patient, args.ckpt, output_dir=output_dir if output_dir else None, show=show_plots)


if __name__ == '__main__':
    main()


