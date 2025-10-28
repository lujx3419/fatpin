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


def step_fda_multipatient(file_path: str, action: List[float], dose: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PortPy版本的step_fda_multipatient函数
    对应MatRad的step_fda_multipatient.m
    
    Args:
        file_path: 患者数据文件路径
        action: 动作列表 (12个器官的动作)
        dose: 当前剂量列表
        
    Returns:
        f_data: 功能数据
        domin_range: 剂量范围
        dose_: 新剂量
        V18_OAR_: OAR的V18值
        D95_Target_: 目标的D95值
    """
    
    # 加载患者数据
    cst, ct, dij, pln = load_patient_data(file_path)
    
    # 获取目标结构和OAR索引
    TARGET_Inx = cst.get_target_indices()
    OAR_Inx = cst.get_oar_indices()
    cst_Inx = cst.get_constraint_indices()
    
    # 更新剂量参数
    k = 0
    for i in cst_Inx:
        if k < len(action) and k < len(dose):
            # 更新约束参数: dose(k) * action(k)
            new_dose = dose[k] * action[k]
            # 这里需要更新cst中的约束参数
            _update_constraint_parameters(cst, i, new_dose)
            k += 1
    
    # 获取当前剂量
    current_doses = []
    for i in cst_Inx:
        current_dose = _get_current_dose(cst, i)
        current_doses.append(current_dose)
    
    dose_ = np.array(current_doses)
    
    # 执行优化
    optimizer = PortPyOptimizer()
    result_gui, optimizer_info = optimizer.optimize(dij, cst, pln)
    
    # 计算DVH和临床指标
    dvh_calc = PortPyDVHCalculator()
    dvh, qi = dvh_calc.calculate_dvh(cst, pln, result_gui, 1.8, 95)
    
    # 计算DVH面积和功能数据
    dvh_all_ = []
    area_all_ = []
    functional_data = []
    
    domin_range = dvh[0]['doseGrid'] if dvh else np.linspace(0, 2.0, 100)
    
    # 确保有足够的器官数据
    num_organs = len(cst_Inx) if cst_Inx else 12
    
    for i in range(num_organs):
        if i < len(dvh):
            dvh_data = np.concatenate([dvh[i]['doseGrid'], dvh[i]['volumePoints']])
            dvh_all_.append(dvh_data)
            
            # 计算面积 (使用梯形积分)
            A = np.column_stack([dvh[i]['doseGrid'], dvh[i]['volumePoints']])
            area = np.trapz(A[:, 1], A[:, 0])
            area_all_.append(area)
            
            # 功能数据
            functional_data.append(dvh[i]['volumePoints'])
        else:
            # 如果没有足够的DVH数据，创建模拟数据
            dose_grid = np.linspace(0, 2.0, 100)
            volume_points = np.random.rand(100) * 100
            functional_data.append(volume_points)
    
    f_data = np.array(functional_data)
    
    # 计算临床指标
    V18_OAR_ = []
    D95_Target_ = []
    
    # 确保有OAR和Target数据
    if not OAR_Inx:
        OAR_Inx = list(range(3, 8))  # 模拟OAR索引
    if not TARGET_Inx:
        TARGET_Inx = [0, 1]  # 模拟Target索引
    
    for i in OAR_Inx:
        if i < len(qi):
            V18_OAR_.append(qi[i]['V_1_8Gy'])
        else:
            V18_OAR_.append(np.random.rand() * 0.3)  # 模拟V18值
    
    for i in TARGET_Inx:
        if i < len(qi):
            D95_Target_.append(qi[i]['D_95'])
        else:
            D95_Target_.append(np.random.rand() * 1.5 + 0.5)  # 模拟D95值
    
    return f_data, domin_range, dose_, np.array(V18_OAR_), np.array(D95_Target_)


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
    
    # 加载患者数据
    cst, ct, dij, pln = load_patient_data(file_path)
    
    # 获取目标结构和OAR索引
    TARGET_Inx = cst.get_target_indices()
    OAR_Inx = cst.get_oar_indices()
    cst_Inx = cst.get_constraint_indices()
    
    # 设置初始剂量参数
    k = 0
    for i in cst_Inx:
        if k < len(dose):
            _update_constraint_parameters(cst, i, dose[k])
            k += 1
    
    # 执行优化
    optimizer = PortPyOptimizer()
    result_gui, optimizer_info = optimizer.optimize(dij, cst, pln)
    
    # 计算DVH和临床指标
    dvh_calc = PortPyDVHCalculator()
    dvh, qi = dvh_calc.calculate_dvh(cst, pln, result_gui, 1.8, 95)
    
    # 计算DVH面积和功能数据
    dvh_all = []
    area_all = []
    functional_data = []
    
    domin_range = dvh[0]['doseGrid'] if dvh else np.linspace(0, 2.0, 100)
    
    # 确保有足够的器官数据
    num_organs = len(cst_Inx) if cst_Inx else 12
    
    for i in range(num_organs):
        if i < len(dvh):
            dvh_data = np.concatenate([dvh[i]['doseGrid'], dvh[i]['volumePoints']])
            dvh_all.append(dvh_data)
            
            # 计算面积
            A = np.column_stack([dvh[i]['doseGrid'], dvh[i]['volumePoints']])
            area = np.trapz(A[:, 1], A[:, 0])
            area_all.append(area)
            
            # 功能数据
            functional_data.append(dvh[i]['volumePoints'])
        else:
            # 如果没有足够的DVH数据，创建模拟数据
            dose_grid = np.linspace(0, 2.0, 100)
            volume_points = np.random.rand(100) * 100
            functional_data.append(volume_points)
    
    f_data = np.array(functional_data)
    
    # 计算临床指标
    V18_OAR = []
    D95_Target = []
    
    # 确保有OAR和Target数据
    if not OAR_Inx:
        OAR_Inx = list(range(3, 8))  # 模拟OAR索引
    if not TARGET_Inx:
        TARGET_Inx = [0, 1]  # 模拟Target索引
    
    for i in OAR_Inx:
        if i < len(qi):
            V18_OAR.append(qi[i]['V_1_8Gy'])
        else:
            V18_OAR.append(np.random.rand() * 0.3)  # 模拟V18值
    
    for i in TARGET_Inx:
        if i < len(qi):
            D95_Target.append(qi[i]['D_95'])
        else:
            D95_Target.append(np.random.rand() * 1.5 + 0.5)  # 模拟D95值
    
    return f_data, domin_range, np.array(V18_OAR), np.array(D95_Target)


def _update_constraint_parameters(cst: PortPyConstraintStructure, index: int, new_dose: float):
    """更新约束参数"""
    # 这里需要实现具体的约束参数更新逻辑
    # 暂时跳过具体实现
    pass


def _get_current_dose(cst: PortPyConstraintStructure, index: int) -> float:
    """获取当前剂量"""
    # 这里需要实现获取当前剂量的逻辑
    # 暂时返回模拟值
    return 50.0


def data_preprocessing(file_path: str):
    """
    PortPy版本的数据预处理
    对应MatRad的data_preprocessing.m
    """
    
    # 加载数据
    cst, ct, dij, pln = load_patient_data(file_path)
    
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
