#!/usr/bin/env python3
"""
PortPy数据结构类 - 对应MatRad中的数据结构
重构自MatRad的cst、ct、dij、pln等结构
"""

import numpy as np
import portpy
from portpy.photon import Plan, Structures, Beams, InfluenceMatrix, CT
from typing import Dict, List, Tuple, Optional, Any
import scipy.io as scio


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
        """获取目标结构索引 (CTV, PTV)"""
        target_indices = []
        for i, (name, info) in enumerate(self.structures.items()):
            if info['type'] in ['CTV', 'PTV']:
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
    计算DVH和临床指标
    """
    
    def __init__(self):
        self.dose_grid = None
        self.volume_points = None
        
    def calculate_dvh(self, cst, pln, result_gui, dose_threshold=1.8, volume_threshold=95):
        """计算DVH和临床指标"""
        dvh_results = []
        qi_results = []
        
        # 模拟DVH计算
        for i, (name, info) in enumerate(cst.structures.items()):
            # 模拟DVH数据
            dose_grid = np.linspace(0, 2.0, 100)
            volume_points = np.random.rand(100) * 100  # 模拟体积百分比
            
            dvh_result = {
                'doseGrid': dose_grid,
                'volumePoints': volume_points
            }
            dvh_results.append(dvh_result)
            
            # 模拟临床指标
            qi_result = {
                'V_1_8Gy': np.random.rand() * 0.5,  # V18
                'D_95': np.random.rand() * 1.5 + 0.5  # D95
            }
            qi_results.append(qi_result)
            
        return dvh_results, qi_results


class PortPyOptimizer:
    """
    对应MatRad中的matRad_fluenceOptimization
    执行通量优化
    """
    
    def __init__(self):
        self.optimizer = None
        
    def optimize(self, dij, cst, pln):
        """执行通量优化"""
        # 模拟优化过程
        result_gui = {
            'dose': np.random.rand(1000),  # 模拟剂量分布
            'fluence': np.random.rand(100)  # 模拟通量分布
        }
        
        optimizer_info = {
            'iterations': 50,
            'converged': True
        }
        
        return result_gui, optimizer_info


def load_patient_data(file_path: str) -> Tuple[PortPyConstraintStructure, Any, Any, PortPyPlan]:
    """
    加载患者数据
    对应MatRad中的load函数
    """
    # 加载.mat文件
    data = scio.loadmat(file_path)
    
    # 创建数据结构
    cst = PortPyConstraintStructure()
    ct = data.get('ct', None)
    dij = data.get('dij', None)
    pln = PortPyPlan()
    
    # 从mat文件加载结构信息
    if 'cst' in data:
        matlab_cst = data['cst']
        # 转换MatRad的cst格式到PortPy格式
        _convert_matlab_cst_to_portpy(matlab_cst, cst)
    
    return cst, ct, dij, pln


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
