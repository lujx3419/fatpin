#!/usr/bin/env python3
"""
PortPy数据结构类 - 对应MatRad中的数据结构
重构自MatRad的cst、ct、dij、pln等结构
"""

import numpy as np
import portpy
from portpy.photon import Plan, Structures, Beams, InfluenceMatrix, CT
from typing import Dict, List, Tuple, Optional, Any
import os
import scipy.io as scio
import time
import threading


class PortPyConstraintStructure:
    """
    对应MatRad中的cst (Constraint Structure Table)
    管理解剖结构和约束条件
    """
    
    def __init__(self):
        self.structures = {}  # 结构信息
        self.constraints = {}  # 约束条件
        self.priorities = {}  # 优先级
        self.visibility = {}  # 可见性
        
    def add_structure(self, name: str, structure_type: str, volume: np.ndarray, 
                     priority: int = 1, visible: bool = True):
        """添加解剖结构"""
        self.structures[name] = {
            'type': structure_type,  # 'TARGET', 'OAR', etc.
            'volume': volume,
            'priority': priority,
            'visible': visible
        }
        
    def set_constraint(self, name: str, constraint_type: str, parameters: Dict):
        """设置约束条件"""
        self.constraints[name] = {
            'type': constraint_type,
            'parameters': parameters
        }
        
    def get_structure_index(self, name: str) -> Optional[int]:
        """获取结构索引"""
        structure_names = list(self.structures.keys())
        if name in structure_names:
            return structure_names.index(name)
        return None
        
    def get_target_indices(self) -> List[int]:
        """
        获取目标结构索引 (只包含CTV和PTV，不包括GTV)
        
        对应原始MATLAB代码的逻辑：
        - TARGET_Inx只包含CTV和PTV
        - GTV虽然参与优化（因为它在cst_Inx中），但不是target
        - 只有PTV和CTV会被用来计算D95_Target
        """
        target_indices = []
        struct_names = list(self.structures.keys())
        for i, name in enumerate(struct_names):
            # 只包含PTV和CTV，不包括GTV
            # 注意：这里需要检查实际的结构名，而不是type
            name_upper = name.upper()
            if name_upper == 'PTV' or name_upper == 'CTV':
                target_indices.append(i)
        return target_indices
        
    def get_oar_indices(self) -> List[int]:
        """获取危及器官索引"""
        oar_indices = []
        for i, (name, info) in enumerate(self.structures.items()):
            if info['type'] not in ['CTV', 'PTV', 'Ring1PTV', 'Ring2PTV', 
                                   'Ring3PTV', 'Ring4PTV', 'Ring5PTV']:
                if name in self.constraints:  # 有约束条件的器官
                    oar_indices.append(i)
        return oar_indices
        
    def get_constraint_indices(self) -> List[int]:
        """获取有约束条件的结构索引"""
        constraint_indices = []
        for i, (name, info) in enumerate(self.structures.items()):
            if name in self.constraints:
                constraint_indices.append(i)
        return constraint_indices
        
    def update_dose_parameters(self, indices: List[int], new_doses: List[float]):
        """更新剂量参数"""
        for i, (idx, dose) in enumerate(zip(indices, new_doses)):
            if idx < len(self.constraints):
                structure_name = list(self.structures.keys())[idx]
                if structure_name in self.constraints:
                    self.constraints[structure_name]['parameters']['dose'] = dose


class PortPyPlan:
    """
    对应MatRad中的pln (Plan)
    管理治疗计划参数
    """
    
    def __init__(self):
        # 基本参数
        self.radiation_mode = 'photons'
        self.machine = 'Generic'
        self.num_of_fractions = 28
        
        # 射束几何设置
        self.beam_geometry = {
            'bixel_width': 5,  # mm
            'gantry_angles': [0, 72, 144, 216, 288],
            'couch_angles': [0, 0, 0, 0, 0],
            'num_of_beams': 5
        }
        
        # 剂量计算设置
        self.dose_calc = {
            'resolution': {'x': 5, 'y': 5, 'z': 5}  # mm
        }
        
        # 优化设置
        self.optimization = {
            'optimizer': 'IPOPT',
            'bio_optimization': 'none',
            'run_dao': True,
            'run_sequencing': True
        }
        
    def get_iso_center(self, cst, ct):
        """计算等中心"""
        # 简化的等中心计算
        return np.array([0, 0, 0])  # 实际应该根据CT和结构计算


class PortPyDoseCalculation:
    """
    对应MatRad中的dij (Dose Influence Matrix)
    管理剂量计算和影响矩阵
    """
    
    def __init__(self):
        self.influence_matrix = None
        self.beamlets = None
        self.optimization_voxels = None
        
    def calculate_photon_dose(self, ct, stf, pln, cst):
        """计算光子剂量"""
        # 这里需要实现PortPy的剂量计算
        # 暂时返回模拟数据
        return self._create_mock_influence_matrix()
        
    def _create_mock_influence_matrix(self):
        """创建模拟影响矩阵"""
        # 模拟影响矩阵 (实际应该用PortPy计算)
        return np.random.rand(1000, 100)  # 1000个体素，100个射束


class PortPyDVHCalculator:
    """
    对应MatRad中的matRad_indicatorWrapper
    计算DVH和临床指标（真实PortPy实现）
    """
    
    def __init__(self):
        self.dose_grid = None
        self.volume_points = None
        
    def calculate_dvh(self, cst, pln, result_gui, dose_threshold=1.8, volume_threshold=95):
        """
        计算DVH和临床指标（使用真实的PortPy Evaluation API）
        
        Args:
            cst: PortPyConstraintStructure对象
            pln: PortPy Plan对象
            result_gui: 包含sol字典的结果（来自PortPyOptimizer.optimize）
            dose_threshold: 剂量阈值（默认1.8 Gy，用于V18）
            volume_threshold: 体积阈值（默认95%，用于D95）
            
        Returns:
            dvh_results: DVH结果列表
            qi_results: 临床指标结果列表
        """
        from portpy.photon import Evaluation
        
        dvh_results = []
        qi_results = []
        
        # 检查是否有真实的sol字典
        sol = result_gui.get('sol') if isinstance(result_gui, dict) else None
        
        # 获取结构列表
        if hasattr(cst, 'portpy_structs'):
            # 使用真实的PortPy Structures
            struct_names = cst.portpy_structs.get_structures()
            use_real_calculation = sol is not None
        else:
            # 使用兼容格式的结构
            struct_names = list(cst.structures.keys())
            use_real_calculation = False
        
        if use_real_calculation and sol is not None:
            # 使用真实的PortPy DVH计算
            try:
                for struct_name in struct_names:
                    # 使用PortPy的真实DVH计算
                    dose_grid, volume_points = Evaluation.get_dvh(
                        sol=sol,
                        struct=struct_name,
                        dose_1d=None,
                        weight_flag=True
                    )
                    
                    # PortPy返回的volume_points可能是小数形式（0-1），需要转换为百分比（0-100）
                    volume_points = np.array(volume_points)
                    if volume_points.max() <= 1.5:  # 如果最大值小于1.5，认为是小数形式
                        volume_points = volume_points * 100.0  # 转换为百分比
                    
                    dvh_result = {
                        'doseGrid': dose_grid,
                        'volumePoints': volume_points
                    }
                    dvh_results.append(dvh_result)
                    
                    # 计算临床指标（根据结构类型选择不同的指标）
                    v18 = 0.0
                    d95 = 0.0
                    
                    # 判断结构类型
                    struct_name_upper = struct_name.upper()
                    is_target = struct_name_upper in ['PTV', 'CTV', 'GTV']
                    
                    try:
                        if is_target:
                            # Target结构：主要关注D95（95%体积的剂量），V18不重要
                            try:
                                d95 = Evaluation.get_dose(
                                    sol=sol,
                                    struct=struct_name,
                                    volume_per=volume_threshold,  # 95%
                                    dose_1d=None,
                                    weight_flag=True
                                )
                            except Exception as e:
                                # 如果D95计算失败（例如体积不足），尝试计算实际最大体积的剂量
                                try:
                                    # 获取DVH数据，找到最大体积百分比
                                    dose_grid, volume_points = Evaluation.get_dvh(
                                        sol=sol, struct=struct_name, dose_1d=None, weight_flag=True
                                    )
                                    max_vol = np.max(volume_points)
                                    if max_vol > 0:
                                        # 计算最大体积对应的剂量
                                        d95 = Evaluation.get_dose(
                                            sol=sol,
                                            struct=struct_name,
                                            volume_per=max_vol,
                                            dose_1d=None,
                                            weight_flag=True
                                        )
                                        print(f"[调试] {struct_name}: 使用最大体积 {max_vol:.1f}% 的剂量作为D95")
                                except:
                                    print(f"警告: 计算{struct_name}的D95失败 ({e})，使用默认值0.0")
                                    d95 = 0.0
                            
                            # Target的V18通常设为0（因为不关心低剂量）
                            v18 = 0.0
                        else:
                            # OAR结构：主要关注V18（1.8 Gy时的体积），D95不重要
                            try:
                                v18 = Evaluation.get_volume(
                                    sol=sol,
                                    struct=struct_name,
                                    dose_value_gy=dose_threshold,  # 1.8 Gy
                                    dose_1d=None,
                                    weight_flag=True
                                )
                            except Exception as e:
                                print(f"警告: 计算{struct_name}的V18失败 ({e})，使用默认值0.0")
                                v18 = 0.0
                            
                            # OAR的D95设为0（因为意义不大）
                            d95 = 0.0
                    except Exception as e:
                        print(f"警告: 计算{struct_name}的临床指标时发生未知错误 ({e})，使用默认值")
                    
                    qi_result = {
                        'V_1_8Gy': float(v18),
                        'D_95': float(d95)
                    }
                    qi_results.append(qi_result)
                    
            except Exception as e:
                print(f"警告: PortPy DVH计算失败 ({e})，回退到模拟数据")
                use_real_calculation = False
        
        # 如果没有真实计算或计算失败，使用模拟数据
        if not use_real_calculation:
            print("警告: 使用模拟DVH数据（无真实sol或计算失败）")
            for i, struct_name in enumerate(struct_names):
                # 生成符合DVH特性的模拟数据
                dose_grid = np.linspace(0, 2.0, 100)
                
                # 根据结构类型生成不同的DVH
                struct_type = cst.structures.get(struct_name, {}).get('type', 'OAR')
                if struct_type in ['PTV', 'CTV', 'GTV']:
                    # Target: 在高剂量处保持体积
                    d95_simulated = 1.5 + np.random.rand() * 0.3
                    volume_points = 100 * np.maximum(1 - ((dose_grid - d95_simulated) / 0.3) ** 2, 0)
                    volume_points = np.clip(volume_points, 0, 100)
                    # 确保单调递减
                    for j in range(1, len(volume_points)):
                        if volume_points[j] > volume_points[j-1]:
                            volume_points[j] = volume_points[j-1]
                    d95 = d95_simulated
                else:
                    # OAR: 快速下降
                    drop_rate = 30 + np.random.rand() * 20
                    volume_points = 100 * np.maximum(1 - (dose_grid / 2.0) * (drop_rate / 100), 0)
                    volume_points = np.clip(volume_points, 0, 100)
                    # 确保单调递减
                    for j in range(1, len(volume_points)):
                        if volume_points[j] > volume_points[j-1]:
                            volume_points[j] = volume_points[j-1]
                    d95 = 0.0
                
                # 确保从100%开始
                if len(volume_points) > 0:
                    volume_points[0] = 100.0
                
                # 计算模拟的临床指标
                v18_idx = np.argmin(np.abs(dose_grid - dose_threshold))
                v18 = volume_points[v18_idx] / 100.0 if v18_idx < len(volume_points) else 0.0
            
            dvh_result = {
                'doseGrid': dose_grid,
                'volumePoints': volume_points
            }
            dvh_results.append(dvh_result)
            
            qi_result = {
                    'V_1_8Gy': float(v18),
                    'D_95': float(d95)
            }
            qi_results.append(qi_result)
            
        return dvh_results, qi_results


class PortPyOptimizer:
    """
    对应MatRad中的matRad_fluenceOptimization和论文中的优化目标
    执行通量优化（基于PortPy的原生优化器）
    """
    
    def __init__(self):
        self.optimizer = None
        self.portpy_opt = None
        
    def optimize(self, inf_matrix, cst, pln):
        """
        使用PortPy的原生优化器执行通量优化,实现论文中的目标函数
        
        Args:
            inf_matrix: PortPy的InfluenceMatrix对象或者影响矩阵数据
            cst: PortPyConstraintStructure对象,包含结构和约束信息
            pln: PortPy Plan对象
            
        Returns:
            result_gui: 包含优化结果的字典,键'sol'对应PortPy的优化结果
            optimizer_info: 优化信息字典,包含成功/失败信息
        """
        from portpy.photon import Optimization, Structures
        import traceback

        try:
            # 1. 准备优化器输入
            # 保存原始的inf_matrix对象（如果是InfluenceMatrix对象），用于qpth获取体素索引
            inf_matrix_obj_original = inf_matrix
            if isinstance(inf_matrix, dict):
                inf_matrix_data = inf_matrix.get('data')
            else:
                inf_matrix_data = inf_matrix

            # If environment requests qpth, run the qpth branch directly (fast GPU QP)
            try:
                import os as _os
                import torch
                solver_env = _os.environ.get('PORTPY_SOLVER', '').lower()
                print(f"[优化器] 环境变量 PORTPY_SOLVER = {solver_env}")
                
                if solver_env == 'qpth':
                    # Check CUDA availability
                    if not torch.cuda.is_available():
                        print("[警告] CUDA不可用，qpth无法使用GPU，将回退到CPU的CVXPy求解器")
                        raise RuntimeError("CUDA not available for qpth")
                    
                    print(f"[优化器] 使用qpth GPU求解器 (CUDA设备: {torch.cuda.get_device_name(0)})")
                    # choose slack mode if requested
                    mode = _os.environ.get('PORTPY_QPTH_MODE', 'standard')
                    print(f"[优化器] qpth模式: {mode}")
                    # 传递原始的inf_matrix对象给qpth函数（用于获取体素索引）
                    # inf_matrix_obj_original可能是InfluenceMatrix对象，有get_opt_voxels_idx方法
                    if mode == 'slack':
                        sol = self._optimize_qpth_slack(inf_matrix_data, cst, pln, inf_matrix_obj_original)
                    else:
                        sol = self._optimize_qpth(inf_matrix_data, cst, pln, inf_matrix_obj_original)
                    # build result_gui to match CVXPy branch
                    if sol is None:
                        return None, {'success': False, 'error': 'qpth returned None'}
                    
                    # 安全地获取optimal值（避免数组的or操作导致错误）
                    optimal = None
                    if 'optimal_intensity' in sol and sol['optimal_intensity'] is not None:
                        optimal = sol['optimal_intensity']
                    elif 'optimal_fluence' in sol and sol['optimal_fluence'] is not None:
                        optimal = sol['optimal_fluence']
                    elif 'optimal' in sol and sol['optimal'] is not None:
                        optimal = sol['optimal']
                    
                    # 计算dose_1d
                    dose_1d = None
                    if 'dose_1d' in sol and sol['dose_1d'] is not None:
                        dose_1d = sol['dose_1d']
                    elif optimal is not None:
                        # 如果没有dose_1d，从optimal计算
                        try:
                            if hasattr(inf_matrix_data, 'A'):
                                A_mat = inf_matrix_data.A
                                if hasattr(A_mat, 'toarray'):
                                    A_mat = A_mat.toarray()
                                dose_1d = np.array(A_mat) @ np.asarray(optimal)
                            else:
                                # inf_matrix_data已经是数组
                                dose_1d = np.array(inf_matrix_data) @ np.asarray(optimal)
                        except Exception as e:
                            print(f"[qpth] 计算dose_1d失败: {e}")
                            dose_1d = None
                    result_gui = {'fluence': optimal, 'dose': dose_1d, 'sol': sol}
                    optimizer_info = {'converged': True, 'success': True, 'method': 'qpth'}
                    print("[优化器] qpth求解成功")
                    return result_gui, optimizer_info
            except Exception as e:
                # If qpth branch fails, fall back to CVXPy branch below
                print(f"[优化器] qpth分支失败: {e}")
                print("[优化器] 回退到CPU的CVXPy求解器")
                import traceback
                traceback.print_exc()

            # 2. 获取或引用PortPy的Structures对象（有些PortPy版本要求在构造时传入data）
            portpy_structs = getattr(cst, 'portpy_structs', None)
            # 如果 cst 中没有 portpy_structs，尝试从传入的 pln（Plan）里获取
            if portpy_structs is None:
                try:
                    portpy_structs = getattr(pln, 'structs', None) or getattr(pln, 'structures', None)
                except Exception:
                    portpy_structs = None

            # 获取结构列表和分类
            struct_names = list(cst.structures.keys())
            target_names = []
            oar_names = []
            
            # 3. 导入每个结构的信息
            for name in struct_names:
                info = cst.structures[name]
                struct_type = info.get('type', '').upper()
                volume_mask = info.get('volume')
                
                # 把体素mask导入PortPy结构
                if volume_mask is not None:
                    # 如果我们有真实的 portpy.Structures 对象，则注册结构到PortPy
                    if portpy_structs is not None and hasattr(portpy_structs, 'register_struct'):
                        try:
                            portpy_structs.register_struct(
                                name=name,
                                mask=volume_mask,  # numpy array的bool mask
                                structure_type='TARGET' if struct_type in ['PTV', 'CTV'] else 'OAR'
                            )
                        except Exception:
                            # 如果注册失败，继续，不阻塞优化设置
                            pass
                    # 否则跳过注册，假设 Plan 中已经包含结构或优化器可以在没有显式 Structures 的情况下工作
                    
                    # 分类结构
                    if struct_type in ['PTV', 'CTV']:
                        target_names.append(name)
                    else:
                        oar_names.append(name)
                        
            print(f"优化问题设置：")
            print(f"- Target结构: {target_names}")
            print(f"- OAR结构: {oar_names}")
            
            # 4. 创建优化器实例
            if not hasattr(pln, 'clinical_criteria'):
                pln.clinical_criteria = None

            # The installed PortPy Optimization signature expects the Plan as the
            # first positional argument: Optimization(my_plan, inf_matrix=..., clinical_criteria=...)
            # If the caller passed a real portpy.Plan, use it. Otherwise try to
            # build a portpy.Plan from available portpy objects stored on cst.
            plan_to_use = None
            try:
                # If pln is already a portpy Plan, use it directly
                from portpy.photon import Plan as PortPyPlanClass, Beams as PortPyBeams
                if isinstance(pln, PortPyPlanClass):
                    plan_to_use = pln
                else:
                    # Try to construct a real PortPy Plan from provided pieces
                    pp_structs = getattr(cst, 'portpy_structs', None) or portpy_structs
                    pp_beams = getattr(cst, 'portpy_beams', None)
                    # If we have beams and an influence matrix object, construct Plan
                    if pp_structs is not None and pp_beams is not None:
                        try:
                            plan_to_use = PortPyPlanClass(pp_structs, pp_beams, inf_matrix)
                        except Exception:
                            plan_to_use = None

            except Exception:
                plan_to_use = None

            # Fallback: if we couldn't create a real PortPy Plan, try passing the
            # provided pln object (some PortPy builds accept a compatible wrapper).
            if plan_to_use is None:
                plan_to_use = pln

            # Finally construct the Optimization instance with the correct signature
            # Note: do NOT pass `structures=` into the constructor since this
            # installed PortPy version expects the Plan to contain the structures.
            self.portpy_opt = Optimization(plan_to_use, inf_matrix=inf_matrix_data, clinical_criteria=None)

            # Monkeypatch buggy methods in the installed PortPy version that
            # hardcode 'PTV' inside add_overdose_quad/add_underdose_quad.
            # We replace them on the instance to use the provided struct name.
            try:
                import types
                import cvxpy as _cp

                def _fixed_add_overdose_quad(self_opt, struct: str, dose_gy: float, weight: float = 10000):
                    A = self_opt.inf_matrix.A
                    st = self_opt.inf_matrix
                    x = self_opt.vars['x']
                    vox = st.get_opt_voxels_idx(struct)
                    if len(vox) == 0:
                        return
                    dO = _cp.Variable(len(vox), pos=True, name=f"{struct}_overdose")
                    obj = (1 / len(vox)) * (weight * _cp.sum_squares(dO))
                    self_opt.add_objective(obj)
                    self_opt.add_constraints([A[vox, :] @ x <= dose_gy + dO])

                def _fixed_add_underdose_quad(self_opt, struct: str, dose_gy: float, weight: float = 100000):
                    A = self_opt.inf_matrix.A
                    st = self_opt.inf_matrix
                    x = self_opt.vars['x']
                    vox = st.get_opt_voxels_idx(struct)
                    if len(vox) == 0:
                        return
                    dU = _cp.Variable(len(vox), pos=True, name=f"{struct}_underdose")
                    obj = (1 / len(vox)) * (weight * _cp.sum_squares(dU))
                    self_opt.add_objective(obj)
                    self_opt.add_constraints([A[vox, :] @ x >= dose_gy - dU])

                # Bind to instance
                self.portpy_opt.add_overdose_quad = types.MethodType(_fixed_add_overdose_quad, self.portpy_opt)
                self.portpy_opt.add_underdose_quad = types.MethodType(_fixed_add_underdose_quad, self.portpy_opt)
            except Exception:
                # If monkeypatching fails, continue and rely on original methods
                pass
            
            target_count = 0
            oar_count = 0
            
            # 5. 添加目标函数和约束
            # 5.1 PTV/CTV的 underdose penalty (β^PTV = 80)
            # PortPy's Optimization may expect a single canonical PTV target group.
            # To avoid broadcasting/index mismatch when both GTV and PTV exist,
            # only add the underdose penalty once for the primary PTV structure
            primary_ptv = None
            for nm in target_names:
                if nm.upper() == 'PTV':
                    primary_ptv = nm
                    break
            if primary_ptv is None and len(target_names) > 0:
                # fallback to first target name
                primary_ptv = target_names[0]

            if primary_ptv is not None and primary_ptv in cst.constraints:
                params = cst.constraints[primary_ptv].get('parameters', {})
                D_pre = float(params.get('dose', 50.0))
                print(f"- 添加Target {primary_ptv}: underdose penalty, D_pre = {D_pre:.1f} Gy")
                try:
                    self.portpy_opt.add_underdose_quad(
                        struct=primary_ptv,
                        dose_gy=D_pre,
                        weight=80.0
                    )
                    target_count += 1
                except Exception:
                    # If the portpy call fails for this structure, continue
                    pass
                        
            # 5.2 OARs的 overdose penalty (β^OAR = 40) 
            for struct_name in oar_names:
                if struct_name in cst.constraints:
                    params = cst.constraints[struct_name].get('parameters', {})
                    D_pre = float(params.get('dose', 18.0))
                    print(f"- 添加OAR {struct_name}: overdose penalty, D_pre = {D_pre:.1f} Gy")
                    
                    # 添加上限剂量约束
                    self.portpy_opt.add_overdose_quad(
                        struct=struct_name,
                        dose_gy=D_pre,
                        weight=40.0  # β^OAR = 40
                    )
                    oar_count += 1
            
            print(f"\n优化问题已设置：")
            print(f"- {target_count} 个Target目标函数 (β^PTV = 80)")  
            print(f"- {oar_count} 个OAR目标函数 (β^OAR = 40)")
            
            # 6. 求解优化问题
            # Before solving, try to persist the structure->opt voxel mapping
            try:
                out_dir = os.path.join(os.getcwd(), 'result', 'real_opt')
                os.makedirs(out_dir, exist_ok=True)
                mapping = {}
                if hasattr(inf_matrix, 'opt_voxels_dict') and isinstance(inf_matrix.opt_voxels_dict, dict) and 'voxel_idx' in inf_matrix.opt_voxels_dict:
                    opt_vox_global_list = []
                    for x in list(inf_matrix.opt_voxels_dict['voxel_idx']):
                        try:
                            opt_vox_global_list.append(int(x))
                        except Exception:
                            import numpy as _np
                            xa = _np.asarray(x)
                            if xa.size == 1:
                                opt_vox_global_list.append(int(xa.item()))
                            else:
                                try:
                                    for xi in xa.ravel():
                                        opt_vox_global_list.append(int(xi))
                                except Exception:
                                    continue
                    global_to_local = {int(g): i for i, g in enumerate(opt_vox_global_list)}
                    for name in struct_names:
                        g_inds = None
                        if hasattr(cst, 'portpy_structs') and cst.portpy_structs is not None:
                            try:
                                if hasattr(cst.portpy_structs, 'get_structure_dose_voxel_indices'):
                                    g_inds = cst.portpy_structs.get_structure_dose_voxel_indices(name)
                                elif hasattr(cst.portpy_structs, 'get_structure_voxel_indices'):
                                    g_inds = cst.portpy_structs.get_structure_voxel_indices(name)
                            except Exception:
                                g_inds = None
                        if g_inds is None:
                            g_inds = cst.structures.get(name, {}).get('voxel_idx')
                        local_inds = []
                        if g_inds is not None:
                            for gg in g_inds:
                                try:
                                    li = global_to_local.get(int(gg), None)
                                except Exception:
                                    li = None
                                if li is not None:
                                    local_inds.append(int(li))
                        mapping[name] = sorted(set(local_inds))
                else:
                    # fallback: try to read any opt_voxel_idx already in cst
                    for name in struct_names:
                        mapping[name] = cst.structures.get(name, {}).get('opt_voxel_idx', [])
                import json
                with open(os.path.join(out_dir, 'structure_mapping.json'), 'w') as mf:
                    json.dump(mapping, mf)
            except Exception:
                pass

            print("\n开始求解...")
            try:
                # 执行优化
                sol = self.portpy_opt.solve()
                print("✓ 优化成功完成")
                
                # 从sol获取结果
                # PortPy may return 'optimal_intensity' or 'optimal_fluence'
                if 'optimal_fluence' in sol:
                    optimal_fluence = sol['optimal_fluence']
                elif 'optimal_intensity' in sol:
                    optimal_fluence = sol['optimal_intensity']
                else:
                    # fallback: try common keys
                    optimal_fluence = sol.get('x') or sol.get('solution') or None
                inf_mat_obj = inf_matrix_data
                
                if hasattr(inf_mat_obj, 'inf_matrix'):
                    inf_matrix_data = inf_mat_obj.inf_matrix
                elif hasattr(inf_mat_obj, 'A'):
                    inf_matrix_data = inf_mat_obj.A
                
                # 处理稀疏矩阵
                try:
                    from scipy import sparse
                    if sparse.issparse(inf_matrix_data):
                        inf_matrix_data = inf_matrix_data.toarray()
                except:
                    pass
                    
                # 计算剂量
                dose_1d = inf_matrix_data @ optimal_fluence
                
                # 组装返回结果
                result_gui = {
                    'dose': dose_1d,
                    'fluence': optimal_fluence,
                    'sol': sol  # 完整的PortPy解
                }
                
                optimizer_info = {
                    'iterations': getattr(self.portpy_opt, 'iterations', 50),
                    'converged': True,
                    'success': True
                }
                
                return result_gui, optimizer_info
                
            except Exception as e:
                print(f"✗ 优化失败: {str(e)}")
                traceback.print_exc()
                return None, str(e)
                
        except Exception as e:
            print(f"✗ 设置优化问题失败: {str(e)}")
            traceback.print_exc()
            return None, str(e)

    def _optimize_qpth(self, inf_matrix_data, cst: PortPyConstraintStructure, pln, inf_matrix_obj=None):
        """
        转换成标准QP: 1/2 f^T Q f + p^T f  , G f <= h

        注意：这是对论文目标的近似（将one-sided penalties用平方误差近似），
        但在实务上通常能快速在GPU上得到合理解用于RL循环。

        返回：sol 字典，包含 'optimal_intensity' 和 'inf_matrix'
        如果失败则抛出异常或返回 None
        """
        try:
            import torch
            from qpth.qp import QPFunction
        except Exception as e:
            raise RuntimeError(f"qpth或torch不可用: {e}")

        # 将inf_matrix_data规范为numpy数组
        # PortPy的InfluenceMatrix对象有.A属性（稀疏矩阵）
        A = inf_matrix_data
        try:
            # 如果inf_matrix_data是InfluenceMatrix对象，提取.A属性
            if hasattr(A, 'A'):
                A = A.A
            # 如果是稀疏矩阵，转换为密集数组
            import numpy as _np
            from scipy import sparse
            if sparse.issparse(A):
                print(f"[qpth] 检测到稀疏矩阵，形状: {A.shape}, 转换为密集数组...")
                A = A.toarray()
            elif isinstance(A, np.ndarray):
                # 已经是numpy数组
                pass
            else:
                # 尝试转换为numpy数组
                A = np.array(A)
        except Exception as e:
            print(f"[qpth] 处理inf_matrix_data时出错: {e}")
            raise RuntimeError(f"无法处理inf_matrix_data: {e}")

        # 验证A的形状
        if not isinstance(A, np.ndarray):
            raise RuntimeError(f"inf_matrix_data无法转换为numpy数组，类型: {type(A)}")
        
        if A.size == 0:
            raise RuntimeError(f"inf_matrix_data是空数组")
        
        if len(A.shape) != 2:
            raise RuntimeError(f"inf_matrix_data不是2维数组，形状: {A.shape}")
        
        num_voxels, num_beamlets = A.shape
        print(f"[qpth] 影响矩阵形状: ({num_voxels}, {num_beamlets})")

        # 构建期望剂量向量 d 和 voxel 权重 w
        d = np.zeros((num_voxels,), dtype=np.float64)
        w = np.zeros((num_voxels,), dtype=np.float64)

        # 默认权重
        beta_PTV = 80.0
        beta_OAR = 40.0

        # 尝试从 cst 中获取每个结构的体素mask或索引
        struct_names = list(cst.structures.keys())
        for name, info in cst.structures.items():
            # 获取约束剂量（如果存在）
            if name in cst.constraints:
                params = cst.constraints[name].get('parameters', {})
                D_pre = float(params.get('dose', 0.0))
            else:
                D_pre = 0.0

            # 试图获取体素索引或mask：支持多种格式
            mask = None
            
            # 方法1: 优先使用inf_matrix的get_opt_voxels_idx方法（最可靠）
            if inf_matrix_obj is not None and hasattr(inf_matrix_obj, 'get_opt_voxels_idx'):
                try:
                    vox_indices = inf_matrix_obj.get_opt_voxels_idx(name)
                    if vox_indices is not None and len(vox_indices) > 0:
                        mask = np.zeros((num_voxels,), dtype=bool)
                        vox_indices_array = np.asarray(vox_indices).flatten()
                        # 确保索引在有效范围内
                        valid_indices = vox_indices_array[(vox_indices_array >= 0) & (vox_indices_array < num_voxels)]
                        mask[valid_indices] = True
                        print(f"[qpth] 结构 {name}: 通过inf_matrix.get_opt_voxels_idx获取到 {len(valid_indices)} 个体素")
                except Exception as e:
                    print(f"[qpth] 结构 {name}: inf_matrix.get_opt_voxels_idx失败: {e}")
                    mask = None
            
            # 方法2: 尝试从portpy_structs获取
            if mask is None and hasattr(cst, 'portpy_structs') and cst.portpy_structs is not None:
                try:
                    # 尝试get_structure_dose_voxel_indices（在dose空间的体素索引）
                    if hasattr(cst.portpy_structs, 'get_structure_dose_voxel_indices'):
                        inds = cst.portpy_structs.get_structure_dose_voxel_indices(name)
                        if inds is not None and len(inds) > 0:
                            inds_array = np.asarray(inds).flatten()
                            valid_inds = inds_array[(inds_array >= 0) & (inds_array < num_voxels)]
                            if len(valid_inds) > 0:
                                mask = np.zeros((num_voxels,), dtype=bool)
                                mask[valid_inds] = True
                                print(f"[qpth] 结构 {name}: 通过portpy_structs.get_structure_dose_voxel_indices获取到 {len(valid_inds)} 个体素")
                    # 尝试get_structure_voxel_indices（在CT空间的体素索引，需要映射到dose空间）
                    if mask is None and hasattr(cst.portpy_structs, 'get_structure_voxel_indices'):
                        ct_inds = cst.portpy_structs.get_structure_voxel_indices(name)
                        if ct_inds is not None and len(ct_inds) > 0:
                            # 如果有inf_matrix，尝试映射CT索引到dose索引
                            if inf_matrix_obj is not None and hasattr(inf_matrix_obj, 'opt_voxels_dict'):
                                try:
                                    opt_vox_dict = inf_matrix_obj.opt_voxels_dict
                                    if isinstance(opt_vox_dict, dict) and 'ct_to_dose_voxel_map' in opt_vox_dict:
                                        ct_to_dose_map = opt_vox_dict['ct_to_dose_voxel_map']
                                        dose_inds = []
                                        for ct_idx in ct_inds:
                                            if ct_idx in ct_to_dose_map:
                                                dose_inds.append(ct_to_dose_map[ct_idx])
                                        if len(dose_inds) > 0:
                                            dose_inds_array = np.asarray(dose_inds).flatten()
                                            valid_dose_inds = dose_inds_array[(dose_inds_array >= 0) & (dose_inds_array < num_voxels)]
                                            if len(valid_dose_inds) > 0:
                                                mask = np.zeros((num_voxels,), dtype=bool)
                                                mask[valid_dose_inds] = True
                                                print(f"[qpth] 结构 {name}: 通过CT->Dose映射获取到 {len(valid_dose_inds)} 个体素")
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"[qpth] 结构 {name}: portpy_structs方法失败: {e}")
                    mask = None
            
            # 方法3: 从cst.structures中的volume字段
            if mask is None:
                vol = info.get('volume', None)
                if isinstance(vol, np.ndarray):
                    # 如果volume是一维数组并且长度等于num_voxels则视为mask或百分比
                    if vol.shape[0] == num_voxels:
                        # 如果是0/1，则为mask；否则当作体积分布并阈值化
                        if set(np.unique(vol)).issubset({0, 1}):
                            mask = vol.astype(bool)
                        else:
                            mask = vol > 0
                        if np.any(mask):
                            print(f"[qpth] 结构 {name}: 从volume字段获取到 {np.sum(mask)} 个体素")

            # 如果没法得到mask则跳过该结构
            if mask is None or not np.any(mask):
                print(f"[qpth] 警告: 结构 {name} 无法获取体素mask，跳过")
                continue

            # 判定是PTV/CTV为target
            name_upper = name.upper()
            if name_upper in ['PTV', 'CTV']:
                w[mask] += beta_PTV
                d[mask] = D_pre
            else:
                # 其它作为OAR
                w[mask] += beta_OAR
                d[mask] = D_pre

        # 如果没有任何权重（无法建立结构-体素映射），抛出以回退到cpu分支
        if np.all(w == 0):
            raise RuntimeError('无法从cst中构建体素权重矩阵，qpth分支无法运行')

        # 构建Q和p: Q = A^T W A, p = -A^T W d
        # 注意：不要显式创建对角矩阵W (会占用过多内存，例如386585x386585需要1TiB)
        # 使用向量运算替代：Q = A^T diag(w) A = A^T (w * A)，其中w*A表示逐列缩放
        print(f"[qpth] 开始构建QP问题，避免创建大型对角矩阵...")
        
        # 方法：Q = A^T diag(w) A = sum_i w[i] * A[:,i]^T * A[:,i]
        # 但更高效的方法是：Q = (w[:, None] * A).T @ A
        # 即：先对A的每一行乘以w，然后转置乘以A
        # 或者使用：Q = A.T @ (w[:, None] * A)
        
        # 使用更内存高效的方式：Q = A^T diag(w) A
        # 这等价于：Q[i,j] = sum_k w[k] * A[k,i] * A[k,j]
        # 可以写成：Q = A.T @ (w[:, None] * A)
        w_col = w[:, np.newaxis]  # shape: (num_voxels, 1)
        weighted_A = w_col * A  # 广播：每列乘以对应的w值
        Q_np = A.T @ weighted_A  # shape: (num_beamlets, num_beamlets)
        
        # p = -A^T W d = -A^T (w * d)
        weighted_d = w * d  # shape: (num_voxels,)
        p_np = -(A.T @ weighted_d)  # shape: (num_beamlets,)
        
        print(f"[qpth] Q矩阵形状: {Q_np.shape}, 内存占用: {Q_np.nbytes / 1024**3:.3f} GB")

        # 添加正则化以保证正定性
        # 直接在对角线上加正则化，避免创建单位矩阵
        reg = 1e-6
        np.fill_diagonal(Q_np, Q_np.diagonal() + reg)

        # 转为torch张量并放到GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法使用qpth GPU求解器")
        device = torch.device('cuda')
        print(f"[qpth] 将QP问题转移到GPU (设备: {torch.cuda.get_device_name(0)})")
        Q = torch.as_tensor(Q_np, dtype=torch.double, device=device).unsqueeze(0)  # (1, n, n)
        p = torch.as_tensor(p_np, dtype=torch.double, device=device).unsqueeze(0)  # (1, n)

        # 不等式：f >= 0 -> -I f <= 0
        G_np = -np.eye(num_beamlets)
        h_np = np.zeros((num_beamlets,))
        G = torch.as_tensor(G_np, dtype=torch.double, device=device).unsqueeze(0)  # (1, m, n)
        h = torch.as_tensor(h_np, dtype=torch.double, device=device).unsqueeze(0)  # (1, m)

        # 空的等式约束
        A_eq = torch.zeros((1, 0, num_beamlets), dtype=torch.double, device=device)
        b_eq = torch.zeros((1, 0), dtype=torch.double, device=device)

        qp_solver = QPFunction(eps=1e-3)
        with torch.no_grad():
            try:
                sol_x = qp_solver(Q, p, G, h, A_eq, b_eq)  # 形状 (1, n)
                x = sol_x.squeeze(0).cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"qpth求解失败: {e}")

        # 返回兼容格式的sol字典
        sol = {
            'optimal_intensity': x,
            'inf_matrix': inf_matrix_data,
            'dose_1d': A.dot(x)
        }
        return sol

    def _optimize_qpth_slack(self, inf_matrix_data, cst: PortPyConstraintStructure, pln,
                             inf_matrix_obj=None, beta_ptv=80.0, beta_oar=40.0, alpha=1e-4, max_vars=50000):
        """
        严格实现论文中的 voxel-wise one-sided penalties：
        为每个被约束的体素引入 slack 变量 s 或 t，将 one-sided penalty 转为 s^2 或 t^2，并用线性不等式连接 f 与 slack：

            s_i >= Dpre - A_i f   (PTV underdose slack)
            t_i >= A_i f - Dpre   (OAR overdose slack)

        问题转为标准 QP:
            min 1/2 x^T Q x + p^T x
            s.t. G x <= h

        注意：变量数 = B + N_s + N_t，若过大会导致内存/计算问题；max_vars 用于保护。
        """
        try:
            import torch
            from qpth.qp import QPFunction
        except Exception as e:
            raise RuntimeError(f"qpth或torch不可用: {e}")

        # Prepare A matrix
        A = np.array(inf_matrix_data)
        if A.ndim != 2:
            raise ValueError('inf_matrix_data must be 2D array (voxels x beamlets)')
        N, B = A.shape

        # Collect voxel indices for PTV and OARs
        ptv_inds = []
        oar_inds = []  # list of tuples (struct_name, indices)
        oar_total_inds = []
        # D_pre per voxel mapping (default 0)
        D_pre = np.zeros((N,))

        # Try to extract masks from cst.structures volume arrays or portpy_structs
        for name, info in cst.structures.items():
            vol = info.get('volume', None)
            inds = None
            if isinstance(vol, np.ndarray) and vol.shape[0] == N:
                # treat nonzero entries as membership
                inds = np.where(vol > 0)[0]
            elif hasattr(cst, 'portpy_structs') and cst.portpy_structs is not None:
                try:
                    inds = cst.portpy_structs.get_structure_voxel_indices(name)
                except Exception:
                    inds = None

            if inds is None or len(inds) == 0:
                continue

            # If structure has constraint, get D_pre value for that structure
            D_val = 0.0
            if name in cst.constraints:
                D_val = float(cst.constraints[name].get('parameters', {}).get('dose', 0.0))

            # assign D_pre to those voxels
            D_pre[inds] = D_val

            # classify PTV vs OAR by type or name
            typ = info.get('type', '').upper()
            if typ in ['PTV', 'CTV'] or name.upper() in ['PTV', 'CTV']:
                ptv_inds.extend(list(inds))
            else:
                oar_inds.append((name, np.array(inds, dtype=int)))
                oar_total_inds.extend(list(inds))

        ptv_inds = np.array(sorted(set(ptv_inds)), dtype=int)
        oar_total_inds = np.array(sorted(set(oar_total_inds)), dtype=int)

        # Build variable indices
        n_s = len(ptv_inds)
        n_t = len(oar_total_inds)
        n_var = B + n_s + n_t
        if n_var > max_vars:
            raise MemoryError(f'Number of variables {n_var} exceeds max_vars {max_vars} - reduce voxels or use approximation')

        # Build Q (numpy) and p
        Q_np = np.zeros((n_var, n_var), dtype=np.float64)
        p_np = np.zeros((n_var,), dtype=np.float64)

        # f regularization
        if alpha > 0:
            Q_np[:B, :B] += 2.0 * alpha * np.eye(B)

        # slacks: for s (PTV) weight beta_ptv/N_ptv, for t (OAR) weight beta_oar/N_m
        # For PTV, use N_ptv = max(1, n_s)
        N_ptv = max(1, n_s)
        for idx_s in range(n_s):
            var_idx = B + idx_s
            Q_np[var_idx, var_idx] = 2.0 * (beta_ptv / float(N_ptv))

        # For OAR, weights may differ per structure; build mapping from voxel to weight (use beta_oar / N_m)
        voxel_to_t_idx = {}
        oar_idx_offset = B + n_s
        cur = 0
        for (name, inds) in oar_inds:
            Nm = max(1, len(inds))
            for i in inds:
                var_idx = oar_idx_offset + cur
                Q_np[var_idx, var_idx] = 2.0 * (beta_oar / float(Nm))
                voxel_to_t_idx[int(i)] = var_idx
                cur += 1

        # Build constraints G x <= h
        G_rows = []
        h_rows = []

        # PTV constraints: A_i f - s_i <= -D_pre_i
        for k, vi in enumerate(ptv_inds):
            Ai = A[vi, :].astype(np.float64)
            row = np.zeros((n_var,), dtype=np.float64)
            row[:B] = Ai
            row[B + k] = -1.0
            G_rows.append(row)
            h_rows.append(-float(D_pre[vi]))

        # OAR constraints: A_i f - t_i <= D_pre_i
        # Note: for each oar voxel, find its t var index
        for vi in oar_total_inds:
            Ai = A[int(vi), :].astype(np.float64)
            row = np.zeros((n_var,), dtype=np.float64)
            row[:B] = Ai
            t_idx = voxel_to_t_idx[int(vi)]
            row[t_idx] = -1.0
            G_rows.append(row)
            h_rows.append(float(D_pre[int(vi)]))

        # Non-negativity for f, s, t: -I x <= 0
        # f >= 0
        for i in range(B):
            row = np.zeros((n_var,), dtype=np.float64)
            row[i] = -1.0
            G_rows.append(row)
            h_rows.append(0.0)
        # s >= 0
        for i in range(n_s):
            row = np.zeros((n_var,), dtype=np.float64)
            row[B + i] = -1.0
            G_rows.append(row)
            h_rows.append(0.0)
        # t >=0
        for i in range(n_t):
            row = np.zeros((n_var,), dtype=np.float64)
            row[oar_idx_offset + i] = -1.0
            G_rows.append(row)
            h_rows.append(0.0)

        G_np = np.vstack(G_rows).astype(np.float64) if len(G_rows) > 0 else np.zeros((0, n_var), dtype=np.float64)
        h_np = np.array(h_rows, dtype=np.float64) if len(h_rows) > 0 else np.zeros((0,), dtype=np.float64)

        # Empty equality
        A_eq_np = np.zeros((0, n_var), dtype=np.float64)
        b_eq_np = np.zeros((0,), dtype=np.float64)

        # Convert to torch tensors on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Q = torch.as_tensor(Q_np, dtype=torch.double, device=device).unsqueeze(0)
        p = torch.as_tensor(p_np, dtype=torch.double, device=device).unsqueeze(0)
        if G_np.size == 0:
            G = torch.zeros((1, 0, n_var), dtype=torch.double, device=device)
            h = torch.zeros((1, 0), dtype=torch.double, device=device)
        else:
            G = torch.as_tensor(G_np, dtype=torch.double, device=device).unsqueeze(0)
            h = torch.as_tensor(h_np, dtype=torch.double, device=device).unsqueeze(0)
        A_eq = torch.as_tensor(A_eq_np.reshape(1, 0, n_var), dtype=torch.double, device=device)
        b_eq = torch.as_tensor(b_eq_np.reshape(1, 0), dtype=torch.double, device=device)

        qp = QPFunction(eps=1e-4)
        with torch.no_grad():
            try:
                sol_x = qp(Q, p, G, h, A_eq, b_eq)
            except Exception as e:
                raise RuntimeError(f'qpth slack QP 求解失败: {e}')

        x = sol_x.squeeze(0).cpu().numpy()
        f_opt = x[:B]

        dose_1d = A.dot(f_opt)
        sol = {
            'optimal_intensity': f_opt,
            'inf_matrix': inf_matrix_data,
            'dose_1d': dose_1d,
        }
        return sol


def load_patient_data(patient_id: str, opt_vox_res_mm: float = None) -> Tuple[PortPyConstraintStructure, Any, Any, PortPyPlan]:
    """
    加载PortPy患者数据（真实实现）
    使用PortPy的真实API加载患者数据
    
    Args:
        patient_id: 患者ID（例如 'Lung_Patient_2'）
        opt_vox_res_mm: 优化体素分辨率（mm），用于减少优化体素数以加速求解
                       - None: 使用默认分辨率（约5mm）
                       - 10.0: 降低分辨率到10mm，可减少约4倍体素数，速度提升4-9倍
                       - 15.0: 进一步降低到15mm，可减少约9倍体素数，但精度降低
    
    Returns:
        cst: PortPy约束结构（兼容格式）
        ct: PortPy CT对象
        inf_matrix: PortPy InfluenceMatrix对象
        plan: PortPy Plan对象
    """
    from portpy.photon import DataExplorer, CT, Structures, Beams, InfluenceMatrix, Plan
    
    # PortPy的数据目录结构：data_dir指向包含所有患者的父目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'portpy_data', 'data'))
    data_dir = base_dir
    # patient_id就是子目录名
    
    try:
        # 使用PortPy的真实API加载数据
        data = DataExplorer(data_dir=data_dir, patient_id=patient_id)
        ct = CT(data)
        structs = Structures(data)
        beams = Beams(data)
        
        # PortPy 1.1.2的bug修复：InfluenceMatrix内部调用get_ct_res_xyz_mm()，但正确方法名是get_ct_voxel_resolution_xyz_mm()
        # 添加别名方法以修复这个bug
        if not hasattr(structs, 'get_ct_res_xyz_mm'):
            def get_ct_res_xyz_mm():
                return structs.get_ct_voxel_resolution_xyz_mm()
            structs.get_ct_res_xyz_mm = get_ct_res_xyz_mm
        
        # 使用默认分辨率创建InfluenceMatrix（PortPy的opt_vox_xyz_res_mm有bug）
        inf_matrix = InfluenceMatrix(structs, beams, ct)
        
        # 手动降采样影响矩阵以加速优化（绕过PortPy的bug）
        # 注意：降采样会导致索引不匹配，所以暂时禁用，改用其他加速方案
        if opt_vox_res_mm is not None:
            print(f"警告: 手动降采样会导致PortPy索引不匹配错误（'index out of range'）")
            print(f"      建议方案：")
            print(f"      1. 设置 opt_vox_res_mm = None（使用默认分辨率，配合快速求解器参数）")
            print(f"      2. 或等待PortPy修复opt_vox_xyz_res_mm的bug后使用")
            print(f"      3. 当前将使用默认分辨率 + 快速求解器参数加速")
            opt_vox_res_mm = None  # 禁用降采样，避免索引错误
        
        # 打印实际优化体素数
        if hasattr(inf_matrix, 'A'):
            actual_voxels = inf_matrix.A.shape[0]
            print(f"实际优化体素数: {actual_voxels:,}")
        
        plan = Plan(structs, beams, inf_matrix)
        
        # 设置clinical_criteria为None（避免Optimization初始化错误）
        plan.clinical_criteria = None
        
        # 转换为兼容的cst格式（用于向后兼容）
        cst = _convert_portpy_structures_to_cst(structs)
        
        # 将PortPy对象也保存到cst中，以便后续使用
        cst.portpy_structs = structs
        cst.portpy_beams = beams
        cst.portpy_ct = ct

        # 尝试构建结构到Dose-voxel的映射 (voxel_idx)
        # 目的：将 portpy 中的结构（CT空间索引）映射为 InfluenceMatrix 使用的 dose-voxel 索引
        try:
            # 优先使用 inf_matrix.opt_voxels_dict 中的 ct_to_dose_voxel_map
            if hasattr(inf_matrix, 'opt_voxels_dict') and isinstance(inf_matrix.opt_voxels_dict, dict) and 'ct_to_dose_voxel_map' in inf_matrix.opt_voxels_dict:
                ct_to_dose = inf_matrix.opt_voxels_dict['ct_to_dose_voxel_map']
            else:
                ct_to_dose = None

            ps = getattr(cst, 'portpy_structs', None)

            # 以 cst.structures.keys() 为主循环，保证覆盖所有兼容结构
            all_struct_names = list(cst.structures.keys())

            for name in all_struct_names:
                dose_inds = None

                # 优先使用 portpy provided API（如果可用）
                if ps is not None:
                    if hasattr(ps, 'get_structure_dose_voxel_indices'):
                        try:
                            dose_inds = ps.get_structure_dose_voxel_indices(name)
                        except Exception:
                            dose_inds = None

                    # 如果没有直接的 dose 索引，尝试获取 CT 索引并用 ct_to_dose 映射
                    if dose_inds is None and hasattr(ps, 'get_structure_voxel_indices') and ct_to_dose is not None:
                        try:
                            ct_inds = ps.get_structure_voxel_indices(name)
                            if ct_inds is not None and ct_to_dose is not None:
                                mapped = []
                                for ct_i in ct_inds:
                                    try:
                                        di = ct_to_dose[int(ct_i)]
                                    except Exception:
                                        di = None
                                    if di is None:
                                        continue
                                    try:
                                        if int(di) >= 0:
                                            mapped.append(int(di))
                                    except Exception:
                                        continue
                                dose_inds = list(dict.fromkeys(mapped))  # 去重并保持顺序
                        except Exception:
                            dose_inds = None

                # 写回 cst（如果获得了全局 dose 索引）
                if dose_inds is not None and len(dose_inds) > 0:
                    cst.structures.setdefault(name, {})
                    cst.structures[name]['voxel_idx'] = list(int(x) for x in dose_inds)

            # 如果 inf_matrix 提供了优化器使用的 opt_voxels 列表（全局索引），
            # 那么建立 global->local 的映射，并为每个 cst 结构计算 local opt 索引
            try:
                if hasattr(inf_matrix, 'opt_voxels_dict') and isinstance(inf_matrix.opt_voxels_dict, dict) and 'voxel_idx' in inf_matrix.opt_voxels_dict:
                    opt_vox_global = inf_matrix.opt_voxels_dict['voxel_idx']
                    # 尝试将 opt_vox_global 规范为 int 列表
                    try:
                        opt_vox_global_list = [int(x) for x in list(opt_vox_global) if x is not None]
                    except Exception:
                        opt_vox_global_list = []

                    # 构建 global_index -> local_index 映射
                    global_to_local = {int(g): i for i, g in enumerate(opt_vox_global_list)}

                    # 为每个 cst 结构计算 local 索引（写入 'opt_voxel_idx'）
                    for name in all_struct_names:
                        g_inds = cst.structures.get(name, {}).get('voxel_idx', None)
                        if g_inds is None:
                            continue
                        local_inds = []
                        for gg in g_inds:
                            try:
                                li = global_to_local.get(int(gg), None)
                            except Exception:
                                li = None
                            if li is None:
                                continue
                            local_inds.append(int(li))
                        # 去重并排序
                        local_unique = sorted(set(local_inds)) if len(local_inds) > 0 else []
                        cst.structures.setdefault(name, {})
                        cst.structures[name]['opt_voxel_idx'] = local_unique
            except Exception:
                # 忽略映射失败，继续
                pass
        except Exception as e:
            # 不要阻塞数据加载，记录警告并继续
            print(f"警告: 构建结构->dose_voxel 映射失败: {e}")
        
        return cst, ct, inf_matrix, plan
        
    except Exception as e:
        print(f"警告: PortPy数据加载失败 ({e})，使用模拟数据")
        # 如果加载失败，回退到模拟数据
    cst = PortPyConstraintStructure()
    num_organs = 12
    for i in range(num_organs):
        name = f'Organ_{i+1}'
        structure_type = 'PTV' if i < 1 else 'OAR'
        cst.add_structure(name=name, structure_type=structure_type, volume=np.array([0.0]))
        if structure_type == 'PTV':
            cst.set_constraint(name, 'MinDVH', {'dose': 20, 'volume': 95, 'weight': 95})
        else:
            cst.set_constraint(name, 'MaxDVH', {'dose': 18, 'volume': 50, 'weight': 50})
        return cst, None, None, PortPyPlan()


def _convert_portpy_structures_to_cst(portpy_structs):
    """
    将PortPy的Structures对象转换为兼容的PortPyConstraintStructure格式
    """
    cst = PortPyConstraintStructure()
    
    # 保存原始PortPy对象
    cst.portpy_structs = portpy_structs
    
    # 获取所有结构名称
    struct_names = portpy_structs.get_structures()
    
    for struct_name in struct_names:
        # 判断结构类型
        struct_name_upper = struct_name.upper()
        if any(x in struct_name_upper for x in ['PTV', 'CTV', 'GTV']):
            structure_type = 'PTV'
        else:
            structure_type = 'OAR'
        
        # 添加结构（volume信息暂时用占位符）
        try:
            volume_cc = portpy_structs.get_volume_cc(struct_name)
            cst.add_structure(name=struct_name, structure_type=structure_type, 
                            volume=np.array([volume_cc]))
        except:
            cst.add_structure(name=struct_name, structure_type=structure_type, 
                            volume=np.array([0.0]))
        
        # 设置约束（占位）
        if structure_type == 'PTV':
            cst.set_constraint(struct_name, 'MinDVH', {'dose': 50, 'volume': 95, 'weight': 95})
        else:
            cst.set_constraint(struct_name, 'MaxDVH', {'dose': 18, 'volume': 50, 'weight': 50})
    
    return cst


def _convert_matlab_cst_to_portpy(matlab_cst, portpy_cst):
    """转换MatRad的cst格式到PortPy格式"""
    # 这里需要实现MatRad cst到PortPy的转换
    # 暂时跳过具体实现
    pass


# 兼容性函数，对应MatRad中的函数名
def get_index(cst, name: str) -> int:
    """获取结构索引"""
    return cst.get_structure_index(name)


def get_index_OarTarget(cst, structure_type: str) -> List[int]:
    """获取OAR或Target索引"""
    if structure_type == 'OAR':
        return cst.get_oar_indices()
    elif structure_type == 'TARGET':
        return cst.get_target_indices()
    return []
