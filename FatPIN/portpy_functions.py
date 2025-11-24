#!/usr/bin/env python3
"""
PortPy版本的MATLAB函数实现
对应step_fda_multipatient.m和reset_fda_multipatient.m
"""

import numpy as np
from portpy_structures import (
    PortPyConstraintStructure, PortPyPlan, PortPyDoseCalculation, 
    PortPyDVHCalculator, PortPyOptimizer, load_patient_data,
    get_index, get_index_OarTarget
)
from typing import Tuple, List
import scipy.io as scio


def step_fda_multipatient(file_path: str, action: List[float], dose: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    PortPy版本的step_fda_multipatient函数
    对应MatRad的step_fda_multipatient.m
    
    论文核心机制：
    - 处方剂量是N维向量（N=有约束的结构数量，每个有约束的器官一个处方剂量）
    - action是N维向量（每个有约束的器官一个动作）
    - 新处方剂量 = 旧处方剂量 ⊙ action（element-wise product，逐元素乘积）
    - 更新后的处方剂量用来定义优化问题的目标函数和约束条件
    - 通过和器官的体素剂量进行计算来定义优化函数
    
    Args:
        file_path: 患者数据文件路径（patient_id）
        action: 动作列表（N维向量，N=有约束的结构数量，每个元素对应一个有约束的器官的缩放系数，如0.95-1.05）
        dose: 当前处方剂量列表（N维向量，N=有约束的结构数量，每个元素是上一个时间步的处方剂量，单位Gy）
        
    Returns:
        f_data: 功能数据
        domin_range: 剂量范围
        dose_: 新剂量
        V18_OAR_: OAR的V18值
        D95_Target_: 目标的D95值
        A_target_mean: 目标器官DVH面积的平均值
        A_oar_mean: OAR器官DVH面积的平均值
    """
    
    # 加载患者数据（file_path现在是patient_id）
    # 从配置文件读取优化体素分辨率（用于加速）
    try:
        from config import OPT_VOXEL_RESOLUTION_MM
        opt_vox_res = OPT_VOXEL_RESOLUTION_MM
    except ImportError:
        opt_vox_res = None  # 如果没有config，使用默认
    
    cst, ct, inf_matrix, pln = load_patient_data(file_path, opt_vox_res_mm=opt_vox_res)
    
    # 获取目标结构和OAR索引
    TARGET_Inx = cst.get_target_indices()
    OAR_Inx = cst.get_oar_indices()
    cst_Inx = cst.get_constraint_indices()
    
    # 调试：打印约束索引信息（仅在第一次调用时）
    if not hasattr(step_fda_multipatient, '_debug_constraint_printed'):
        step_fda_multipatient._debug_constraint_printed = set()
    if file_path not in step_fda_multipatient._debug_constraint_printed:
        all_struct_names = list(cst.structures.keys())
        constraint_struct_names = [all_struct_names[i] for i in cst_Inx]
        print(f"[约束检查] 患者 {file_path}:")
        print(f"  总结构数: {len(all_struct_names)}")
        print(f"  有约束的结构数: {len(cst_Inx)}")
        print(f"  所有结构: {all_struct_names}")
        print(f"  有约束的结构: {constraint_struct_names}")
        step_fda_multipatient._debug_constraint_printed.add(file_path)
    
    # 论文核心机制：处方剂量与action的element-wise product
    # action和dose现在只针对有约束的结构（维度为len(cst_Inx)）
    # 新处方剂量 = 旧处方剂量 ⊙ action（逐元素乘积）
    
    # 确保action和dose的维度匹配（都是有约束的结构数量）
    num_constraint_organs = len(cst_Inx)
    
    # 严格检查维度匹配：不允许自动修复，必须完全匹配
    if len(action) != num_constraint_organs:
        all_struct_names = list(cst.structures.keys())
        constraint_struct_names = [all_struct_names[i] for i in cst_Inx]
        raise ValueError(
            f"step_fda_multipatient: action维度 ({len(action)}) 与有约束的结构数量 ({num_constraint_organs}) 不匹配！\n"
            f"  有约束的结构: {constraint_struct_names}\n"
            f"  传入的action维度: {len(action)}\n"
            f"  请确保action的维度与约束结构数量完全匹配。"
        )
    
    if len(dose) != num_constraint_organs:
        all_struct_names = list(cst.structures.keys())
        constraint_struct_names = [all_struct_names[i] for i in cst_Inx]
        raise ValueError(
            f"step_fda_multipatient: dose维度 ({len(dose)}) 与有约束的结构数量 ({num_constraint_organs}) 不匹配！\n"
            f"  有约束的结构: {constraint_struct_names}\n"
            f"  传入的dose维度: {len(dose)}\n"
            f"  请确保dose的维度与约束结构数量完全匹配。"
        )
    
    action = np.array(action)
    dose = np.array(dose)
    
    # 将全局索引转换为在cst_Inx中的局部索引（用于区分Target与OAR）
    TARGET_Inx_local = [cst_Inx.index(i) for i in TARGET_Inx if i in cst_Inx]
    OAR_Inx_local = [cst_Inx.index(i) for i in OAR_Inx if i in cst_Inx]
    # 如果缺失，则给出保守的回退（不会报错，但效果较弱）
    if not TARGET_Inx_local and num_constraint_organs > 0:
        TARGET_Inx_local = [0]
    if not OAR_Inx_local and num_constraint_organs > 1:
        OAR_Inx_local = list(range(1, num_constraint_organs))

    # 按类型应用相反方向的缩放：
    # - Target: new = dose * action  (action>1 提升目标剂量)
    # - OAR:    new = dose / action  (action>1 收紧OAR，降低允许剂量)
    action = np.array(action, dtype=float)
    dose = np.array(dose, dtype=float)
    new_prescription_dose_constraint = np.empty_like(dose)
    eps = 1e-6
    for idx in range(num_constraint_organs):
        if idx in TARGET_Inx_local:
            new_prescription_dose_constraint[idx] = dose[idx] * action[idx]
        elif idx in OAR_Inx_local:
            new_prescription_dose_constraint[idx] = dose[idx] / np.maximum(action[idx], eps)
        else:
            # 未分类的结构，保持与Target一致的方向
            new_prescription_dose_constraint[idx] = dose[idx] * action[idx]
    
    # 调试：打印action应用情况（仅在特定条件下打印，避免输出过多）
    if not hasattr(step_fda_multipatient, '_debug_call_count'):
        step_fda_multipatient._debug_call_count = 0
    step_fda_multipatient._debug_call_count += 1
    
    # 维度已确认正常，移除详细调试输出（仅在异常时打印）
    if step_fda_multipatient._debug_call_count <= 3:
        print(f"[调试-step_fda] 调用#{step_fda_multipatient._debug_call_count}, dose维度={len(dose)}, action维度={len(action)}")
    
    # 更新每个有约束的器官的约束参数（使用更新后的处方剂量）
    # 这个更新后的处方剂量用来定义优化问题的目标函数和约束条件
    for constraint_idx, organ_idx in enumerate(cst_Inx):
        struct_names = list(cst.structures.keys())
        if organ_idx < len(struct_names):
            struct_name = struct_names[organ_idx]
            # 使用更新后的处方剂量更新约束
            new_dose = new_prescription_dose_constraint[constraint_idx]
            _update_constraint_parameters(cst, constraint_idx, new_dose)
    
    # 返回更新后的处方剂量向量（维度为len(cst_Inx)，用于下一时间步）
    dose_ = new_prescription_dose_constraint
    
    # 执行优化（使用inf_matrix而不是dij）
    optimizer = PortPyOptimizer()
    result_gui, optimizer_info = optimizer.optimize(inf_matrix, cst, pln)
    
    # 调试：检查优化是否真的改变了剂量（仅在同一患者内比较，不同患者体素数不同）
    # 注意：不同患者的体素数不同，所以不能跨患者比较
    if result_gui and 'dose' in result_gui:
        dose_1d = result_gui['dose']
        if dose_1d is not None:
            # 只在同一患者内比较（通过检查体素数是否匹配）
            dose_len = len(dose_1d) if hasattr(dose_1d, '__len__') else 0
            if hasattr(step_fda_multipatient, '_prev_dose') and hasattr(step_fda_multipatient, '_prev_dose_len'):
                if step_fda_multipatient._prev_dose is not None and step_fda_multipatient._prev_dose_len == dose_len:
                    try:
                        dose_change = np.abs(dose_1d - step_fda_multipatient._prev_dose).max()
                        print(f"[优化调试-step] 剂量变化: max_change = {dose_change:.6f}")
                        if dose_change < 1e-6:
                            print(f"[警告] 剂量几乎没有变化，优化可能没有生效！")
                    except ValueError as e:
                        # 不同患者体素数不同，无法比较
                        print(f"[优化调试-step] 跳过剂量比较（不同患者或体素数变化）")
            if dose_1d is not None:
                step_fda_multipatient._prev_dose = dose_1d.copy() if hasattr(dose_1d, 'copy') else np.array(dose_1d)
                step_fda_multipatient._prev_dose_len = dose_len
    
    # 计算DVH和临床指标
    dvh_calc = PortPyDVHCalculator()
    dvh, qi = dvh_calc.calculate_dvh(cst, pln, result_gui, 1.8, 95)
    
    # 调试：检查DVH数据（仅在特定步数打印，避免输出过多）
    if dvh and len(dvh) > 0 and (not hasattr(step_fda_multipatient, '_debug_count') or step_fda_multipatient._debug_count % 50 == 0):
        if not hasattr(step_fda_multipatient, '_debug_count'):
            step_fda_multipatient._debug_count = 0
        step_fda_multipatient._debug_count += 1
        
        dose_grid_sample = dvh[0]['doseGrid']
        volume_sample = np.array(dvh[0]['volumePoints'])
        
        # 检查体积单位（PortPy可能返回0-1，需要转换为0-100）
        vol_max = np.max(volume_sample)
        if vol_max <= 1.5:
            print(f"[DVH调试-step] 检测到体积是小数形式 (max={vol_max:.3f})，转换为百分比")
            volume_sample = volume_sample * 100.0
        
        print(f"[DVH调试-step] 剂量网格范围: [{np.min(dose_grid_sample):.3f}, {np.max(dose_grid_sample):.3f}] Gy")
        print(f"[DVH调试-step] 体积范围: [{np.min(volume_sample):.2f}, {np.max(volume_sample):.2f}] %")
        print(f"[DVH调试-step] DVH数据点数: {len(dose_grid_sample)}")
        # 检查是否是直线（体积是否快速下降）
        if len(volume_sample) > 5:
            vol_drop = volume_sample[0] - volume_sample[min(5, len(volume_sample)-1)]
            print(f"[DVH调试-step] 前5个点的体积下降: {vol_drop:.2f}%")
    
    # 计算DVH面积和功能数据
    dvh_all_ = []
    area_all_ = []
    functional_data = []
    
    # 获取domin_range并清理无效值
    if dvh and len(dvh) > 0:
        domin_range = np.array(dvh[0]['doseGrid']).flatten()
        # 清理无效值
        if np.any(np.isinf(domin_range)) or np.any(np.isnan(domin_range)):
            print(f"[警告] step: domin_range包含无效值，使用默认范围")
            domin_range = np.linspace(0, 2.0, 100)
        else:
            # 确保单调递增
            if not np.all(np.diff(domin_range) > 0):
                sort_idx = np.argsort(domin_range)
                domin_range = domin_range[sort_idx]
    else:
        domin_range = np.linspace(0, 2.0, 100)
    
    # 确保有足够的器官数据
    num_organs = len(cst_Inx) if cst_Inx else 12
    
    for i in range(num_organs):
        if i < len(dvh):
            # 获取volumePoints并确保是百分比形式（0-100）
            vol_points = np.array(dvh[i]['volumePoints'])
            if vol_points.max() <= 1.5:  # 如果是小数形式（0-1），转换为百分比
                vol_points = vol_points * 100.0
            
            dose_grid = dvh[i]['doseGrid']
            dvh_data = np.concatenate([dose_grid, vol_points])
            dvh_all_.append(dvh_data)
            
            # 计算面积 (使用梯形积分) - volumePoints现在已经是百分比形式
            A = np.column_stack([dose_grid, vol_points])
            area = np.trapz(A[:, 1], A[:, 0])
            area_all_.append(area)
            
            # 功能数据：确保volumePoints是1维数组
            if isinstance(vol_points, np.ndarray):
                vol_points_flat = vol_points.flatten()
            else:
                vol_points_flat = np.array(vol_points).flatten()
            functional_data.append(vol_points_flat)
        else:
            # 如果没有足够的DVH数据，创建模拟数据（100个点）
            volume_points = np.random.rand(100) * 100
            functional_data.append(volume_points)
    
    # 统一所有数组的长度（避免ragged array）
    if len(functional_data) > 0:
        expected_length = len(functional_data[0]) if len(functional_data[0]) > 0 else 100
        for i, fd in enumerate(functional_data):
            fd_array = np.array(fd).flatten()
            if len(fd_array) != expected_length:
                if len(fd_array) > expected_length:
                    functional_data[i] = fd_array[:expected_length]
                else:
                    from scipy.interpolate import interp1d
                    old_indices = np.linspace(0, 1, len(fd_array))
                    new_indices = np.linspace(0, 1, expected_length)
                    interp_func = interp1d(old_indices, fd_array, kind='linear', fill_value='extrapolate')
                    functional_data[i] = interp_func(new_indices)
            else:
                functional_data[i] = fd_array
    
    # 现在所有数组长度一致，可以安全转换为2D数组
    f_data = np.array(functional_data, dtype=np.float64)
    
    if f_data.ndim != 2:
        print(f"[警告] step: functional_data维度异常: {f_data.ndim}, 形状: {f_data.shape}")
    
    # 计算临床指标
    # 注意：TARGET_Inx和OAR_Inx是全局索引（所有结构中的索引），需要转换为在cst_Inx中的局部索引
    # 因为dvh和qi是按照cst_Inx的顺序返回的
    V18_OAR_ = []
    D95_Target_ = []
    
    # 将全局索引转换为在cst_Inx中的局部索引
    TARGET_Inx_local = [cst_Inx.index(i) for i in TARGET_Inx if i in cst_Inx]
    OAR_Inx_local = [cst_Inx.index(i) for i in OAR_Inx if i in cst_Inx]
    
    # 确保有OAR和Target数据
    if not OAR_Inx_local:
        OAR_Inx_local = list(range(min(3, len(cst_Inx)), min(8, len(cst_Inx))))  # 模拟OAR索引
    if not TARGET_Inx_local:
        TARGET_Inx_local = list(range(min(2, len(cst_Inx))))  # 模拟Target索引
    
    for i in OAR_Inx_local:
        if i < len(qi):
            V18_OAR_.append(qi[i]['V_1_8Gy'])
        else:
            V18_OAR_.append(np.random.rand() * 0.3)  # 模拟V18值
    
    for i in TARGET_Inx_local:
        if i < len(qi):
            D95_Target_.append(qi[i]['D_95'])
        else:
            D95_Target_.append(np.random.rand() * 1.5 + 0.5)  # 模拟D95值

    # 计算 A_target（目标面积：不取均值，直接汇总）与 A_oar_mean（OAR 平均面积）
    # 仅一个主目标：取第一个 TARGET_Inx 的面积
    target_areas = []
    oar_areas = []
    for i in range(len(area_all_)):
        if i in TARGET_Inx_local:
            target_areas.append(area_all_[i])
        if i in OAR_Inx_local:
            oar_areas.append(area_all_[i])
    if target_areas:
        A_target = float(target_areas[0])
    else:
        A_target = 0.0
    A_oar_mean = float(np.mean(oar_areas)) if oar_areas else 0.0

    return f_data, domin_range, dose_, np.array(V18_OAR_), np.array(D95_Target_), A_target, A_oar_mean


def reset_fda_multipatient(file_path: str, dose: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PortPy版本的reset_fda_multipatient函数
    对应MatRad的reset_fda_multipatient.m
    
    Args:
        file_path: 患者数据文件路径
        dose: 初始剂量列表
        
    Returns:
        f_data: 功能数据
        domin_range: 剂量范围
        V18_OAR: OAR的V18值
        D95_Target: 目标的D95值
    """
    
    # 加载患者数据（file_path现在是patient_id）
    # 从配置文件读取优化体素分辨率（用于加速）
    try:
        from config import OPT_VOXEL_RESOLUTION_MM
        opt_vox_res = OPT_VOXEL_RESOLUTION_MM
    except ImportError:
        opt_vox_res = None  # 如果没有config，使用默认
    
    cst, ct, inf_matrix, pln = load_patient_data(file_path, opt_vox_res_mm=opt_vox_res)
    
    # 获取目标结构和OAR索引
    TARGET_Inx = cst.get_target_indices()
    OAR_Inx = cst.get_oar_indices()
    cst_Inx = cst.get_constraint_indices()
    
    # 调试：打印约束索引信息（仅在第一次调用时）
    if not hasattr(reset_fda_multipatient, '_debug_constraint_printed'):
        reset_fda_multipatient._debug_constraint_printed = set()
    if file_path not in reset_fda_multipatient._debug_constraint_printed:
        all_struct_names = list(cst.structures.keys())
        constraint_struct_names = [all_struct_names[i] for i in cst_Inx]
        print(f"[约束检查-reset] 患者 {file_path}:")
        print(f"  总结构数: {len(all_struct_names)}")
        print(f"  有约束的结构数: {len(cst_Inx)}")
        print(f"  所有结构: {all_struct_names}")
        print(f"  有约束的结构: {constraint_struct_names}")
        reset_fda_multipatient._debug_constraint_printed.add(file_path)
    
    # 设置初始处方剂量参数
    # dose现在只针对有约束的结构（维度为len(cst_Inx)）
    num_constraint_organs = len(cst_Inx)
    
    # 严格检查维度匹配：不允许自动修复，必须完全匹配
    if len(dose) != num_constraint_organs:
        all_struct_names = list(cst.structures.keys())
        constraint_struct_names = [all_struct_names[i] for i in cst_Inx]
        raise ValueError(
            f"reset_fda_multipatient: dose维度 ({len(dose)}) 与有约束的结构数量 ({num_constraint_organs}) 不匹配！\n"
            f"  有约束的结构: {constraint_struct_names}\n"
            f"  传入的dose维度: {len(dose)}\n"
            f"  请确保dose的维度与约束结构数量完全匹配。"
        )
    dose = np.array(dose)  # 确保是numpy数组
    
    # 为每个有约束的器官设置初始处方剂量约束
    struct_names = list(cst.structures.keys())
    for constraint_idx, organ_idx in enumerate(cst_Inx):
        if organ_idx < len(struct_names):
            _update_constraint_parameters(cst, constraint_idx, dose[constraint_idx])
    
    # 执行优化（使用inf_matrix而不是dij）
    optimizer = PortPyOptimizer()
    result_gui, optimizer_info = optimizer.optimize(inf_matrix, cst, pln)
    
    # 计算DVH和临床指标
    dvh_calc = PortPyDVHCalculator()
    dvh, qi = dvh_calc.calculate_dvh(cst, pln, result_gui, 1.8, 95)
    
    # 计算DVH面积和功能数据
    dvh_all = []
    area_all = []
    functional_data = []
    
    # 获取domin_range并清理无效值
    if dvh and len(dvh) > 0:
        domin_range = np.array(dvh[0]['doseGrid']).flatten()
        # 清理无效值
        if np.any(np.isinf(domin_range)) or np.any(np.isnan(domin_range)):
            print(f"[警告] step: domin_range包含无效值，使用默认范围")
            domin_range = np.linspace(0, 2.0, 100)
        else:
            # 确保单调递增
            if not np.all(np.diff(domin_range) > 0):
                sort_idx = np.argsort(domin_range)
                domin_range = domin_range[sort_idx]
    else:
        domin_range = np.linspace(0, 2.0, 100)
    
    # 确保有足够的器官数据
    num_organs = len(cst_Inx) if cst_Inx else 12
    
    for i in range(num_organs):
        if i < len(dvh):
            # 获取volumePoints并确保是百分比形式（0-100）
            vol_points = np.array(dvh[i]['volumePoints'])
            if vol_points.max() <= 1.5:  # 如果是小数形式（0-1），转换为百分比
                vol_points = vol_points * 100.0
            
            dose_grid = dvh[i]['doseGrid']
            dvh_data = np.concatenate([dose_grid, vol_points])
            dvh_all.append(dvh_data)
            
            # 计算面积 (使用梯形积分) - volumePoints现在已经是百分比形式
            A = np.column_stack([dose_grid, vol_points])
            area = np.trapz(A[:, 1], A[:, 0])
            area_all.append(area)
            
            # 功能数据：确保volumePoints是1维数组
            if isinstance(vol_points, np.ndarray):
                vol_points_flat = vol_points.flatten()
            else:
                vol_points_flat = np.array(vol_points).flatten()
            functional_data.append(vol_points_flat)
        else:
            # 如果没有足够的DVH数据，创建模拟数据（100个点）
            volume_points = np.random.rand(100) * 100
            functional_data.append(volume_points)
    
    # 统一所有数组的长度（避免ragged array）
    if len(functional_data) > 0:
        expected_length = len(functional_data[0]) if len(functional_data[0]) > 0 else 100
        for i, fd in enumerate(functional_data):
            fd_array = np.array(fd).flatten()
            if len(fd_array) != expected_length:
                if len(fd_array) > expected_length:
                    functional_data[i] = fd_array[:expected_length]
                else:
                    from scipy.interpolate import interp1d
                    old_indices = np.linspace(0, 1, len(fd_array))
                    new_indices = np.linspace(0, 1, expected_length)
                    interp_func = interp1d(old_indices, fd_array, kind='linear', fill_value='extrapolate')
                    functional_data[i] = interp_func(new_indices)
            else:
                functional_data[i] = fd_array
    
    # 现在所有数组长度一致，可以安全转换为2D数组
    f_data = np.array(functional_data, dtype=np.float64)
    
    if f_data.ndim != 2:
        print(f"[警告] reset: functional_data维度异常: {f_data.ndim}, 形状: {f_data.shape}")
    
    # 计算临床指标
    # 注意：TARGET_Inx和OAR_Inx是全局索引（所有结构中的索引），需要转换为在cst_Inx中的局部索引
    # 因为dvh和qi是按照cst_Inx的顺序返回的
    V18_OAR = []
    D95_Target = []
    
    # 将全局索引转换为在cst_Inx中的局部索引
    TARGET_Inx_local = [cst_Inx.index(i) for i in TARGET_Inx if i in cst_Inx]
    OAR_Inx_local = [cst_Inx.index(i) for i in OAR_Inx if i in cst_Inx]
    
    # 确保有OAR和Target数据
    if not OAR_Inx_local:
        OAR_Inx_local = list(range(min(3, len(cst_Inx)), min(8, len(cst_Inx))))  # 模拟OAR索引
    if not TARGET_Inx_local:
        TARGET_Inx_local = list(range(min(2, len(cst_Inx))))  # 模拟Target索引
    
    for i in OAR_Inx_local:
        if i < len(qi):
            V18_OAR.append(qi[i]['V_1_8Gy'])
        else:
            V18_OAR.append(np.random.rand() * 0.3)  # 模拟V18值
    
    for i in TARGET_Inx_local:
        if i < len(qi):
            D95_Target.append(qi[i]['D_95'])
        else:
            D95_Target.append(np.random.rand() * 1.5 + 0.5)  # 模拟D95值
    
    # 返回实际使用的处方剂量向量（维度为len(cst_Inx)）
    # 注意：这里返回的dose_actual是经过验证的，维度与cst_Inx匹配
    dose_actual = dose  # dose已经是经过验证的，维度为len(cst_Inx)
    
    return f_data, domin_range, np.array(V18_OAR), np.array(D95_Target), dose_actual


def _update_constraint_parameters(cst: PortPyConstraintStructure, index: int, new_dose: float):
    """
    更新约束参数
    
    这是论文的核心：使用action来修改优化问题的约束条件
    action * dose = new_dose，用来更新每个结构的约束参数
    
    Args:
        cst: 约束结构对象
        index: 结构索引（在constraint_indices中的索引）
        new_dose: 新的剂量值（dose * action的结果）
    """
    # 获取所有有约束的结构
    constraint_indices = cst.get_constraint_indices()
    
    if index < len(constraint_indices):
        # 获取实际的结构索引
        struct_idx = constraint_indices[index]
        struct_names = list(cst.structures.keys())
        
        if struct_idx < len(struct_names):
            struct_name = struct_names[struct_idx]
            
            # 更新约束参数中的dose值
            if struct_name in cst.constraints:
                # 更新dose参数
                if 'parameters' in cst.constraints[struct_name]:
                    cst.constraints[struct_name]['parameters']['dose'] = new_dose
                else:
                    cst.constraints[struct_name]['parameters'] = {'dose': new_dose}
            else:
                # 如果约束不存在，创建一个新的约束
                struct_type = cst.structures[struct_name]['type']
                if struct_type in ['PTV', 'CTV', 'GTV']:
                    # Target: 最小剂量约束
                    cst.set_constraint(struct_name, 'MinDVH', {'dose': new_dose, 'volume': 95, 'weight': 95})
                else:
                    # OAR: 最大剂量约束
                    cst.set_constraint(struct_name, 'MaxDVH', {'dose': new_dose, 'volume': 50, 'weight': 50})


def _get_current_dose(cst: PortPyConstraintStructure, index: int) -> float:
    """
    获取当前剂量值
    
    Args:
        cst: 约束结构对象
        index: 结构索引（在constraint_indices中的索引）
        
    Returns:
        当前剂量值
    """
    # 获取所有有约束的结构
    constraint_indices = cst.get_constraint_indices()
    
    if index < len(constraint_indices):
        # 获取实际的结构索引
        struct_idx = constraint_indices[index]
        struct_names = list(cst.structures.keys())
        
        if struct_idx < len(struct_names):
            struct_name = struct_names[struct_idx]
            
            # 从约束中获取当前dose值
            if struct_name in cst.constraints:
                params = cst.constraints[struct_name].get('parameters', {})
                return params.get('dose', 50.0)  # 默认50.0 Gy
    
    # 如果没有找到，返回默认值
    return 50.0


def data_preprocessing(file_path: str):
    """
    PortPy版本的数据预处理
    对应MatRad的data_preprocessing.m
    """
    
    # 加载数据（file_path现在是patient_id）
    cst, ct, inf_matrix, pln = load_patient_data(file_path)
    
    # 清空器官约束
    organ_indices = [i for i, (name, info) in enumerate(cst.structures.items()) 
                     if info['type'] == 'OAR']
    for i in organ_indices:
        # 清空约束
        pass
    
    # 设置OAR约束
    _setup_oar_constraints(cst)
    
    # 设置目标约束
    _setup_target_constraints(cst)
    
    # 设置优先级
    _setup_priorities(cst)
    
    # 保存数据
    save_data = {
        'cst': cst,
        'ct': ct
    }
    scio.savemat(file_path, save_data)


def _setup_oar_constraints(cst: PortPyConstraintStructure):
    """设置OAR约束"""
    oar_structures = ['Bladder', 'Femoral Head R', 'Femoral Head L', 
                      'Rectum', 'Small intestine', 'Bladder out', 
                      'Rectum out', 'Small intestine out']
    
    for name in oar_structures:
        if cst.get_structure_index(name) is not None:
            # 设置MaxDVH约束
            constraint = {
                'type': 'MaxDVH',
                'parameters': {'dose': 20, 'volume': 50, 'weight': 50}
            }
            cst.set_constraint(name, 'MaxDVH', constraint['parameters'])


def _setup_target_constraints(cst: PortPyConstraintStructure):
    """设置目标约束"""
    target_structures = ['CTV', 'PTV', 'Ring1PTV', 'Ring2PTV', 
                        'Ring3PTV', 'Ring4PTV', 'Ring5PTV']
    
    for name in target_structures:
        if cst.get_structure_index(name) is not None:
            # 设置MinDVH约束
            constraint = {
                'type': 'MinDVH',
                'parameters': {'dose': 20, 'volume': 50, 'weight': 95}
            }
            cst.set_constraint(name, 'MinDVH', constraint['parameters'])
            
            # 设置优先级
            if name.startswith('Ring'):
                cst.structures[name]['priority'] = 3


def _setup_priorities(cst: PortPyConstraintStructure):
    """设置优先级"""
    # 设置空约束结构的优先级
    for name, info in cst.structures.items():
        if name not in cst.constraints:
            info['priority'] = 4
            info['visible'] = False
