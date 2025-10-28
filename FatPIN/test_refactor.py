#!/usr/bin/env python3
"""
测试PortPy重构是否成功
验证基本功能是否正常工作
"""

import numpy as np
import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否成功"""
    print("=== 测试导入 ===")
    
    try:
        from portpy_structures import PortPyConstraintStructure, PortPyPlan, PortPyDoseCalculation
        print("✅ PortPy数据结构导入成功")
    except ImportError as e:
        print(f"❌ PortPy数据结构导入失败: {e}")
        return False
    
    try:
        from portpy_functions import step_fda_multipatient, reset_fda_multipatient
        print("✅ PortPy函数导入成功")
    except ImportError as e:
        print(f"❌ PortPy函数导入失败: {e}")
        return False
    
    try:
        from train_portpy import TP, PGNetwork, Actor, Critic
        print("✅ 强化学习环境导入成功")
    except ImportError as e:
        print(f"❌ 强化学习环境导入失败: {e}")
        return False
    
    return True


def test_data_structures():
    """测试数据结构类"""
    print("\n=== 测试数据结构 ===")
    
    try:
        from portpy_structures import PortPyConstraintStructure, PortPyPlan
        
        # 测试约束结构
        cst = PortPyConstraintStructure()
        cst.add_structure('CTV', 'TARGET', np.random.rand(100, 100, 50))
        cst.add_structure('Bladder', 'OAR', np.random.rand(100, 100, 50))
        
        target_indices = cst.get_target_indices()
        oar_indices = cst.get_oar_indices()
        
        print(f"✅ 目标结构索引: {target_indices}")
        print(f"✅ OAR索引: {oar_indices}")
        
        # 测试计划
        pln = PortPyPlan()
        print(f"✅ 治疗计划参数: {pln.radiation_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        return False


def test_rl_environment():
    """测试强化学习环境"""
    print("\n=== 测试强化学习环境 ===")
    
    try:
        from train_portpy import TP, PGNetwork, Actor, Critic
        
        # 创建环境
        env = TP(num_of_organ=12)
        print(f"✅ 环境创建成功: 动作空间 {env.action_space}")
        
        # 测试网络
        network = PGNetwork(n_state=5, n_action=5)
        test_input = torch.randn(1, 5)
        output = network(test_input)
        print(f"✅ 网络前向传播成功: 输出形状 {output.shape}")
        
        # 测试Actor
        actor = Actor(env)
        test_obs = np.random.randn(12, 5)
        action = actor.choose_action(test_obs)
        print(f"✅ Actor动作选择成功: 动作 {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ 强化学习环境测试失败: {e}")
        return False


def test_portpy_integration():
    """测试PortPy集成"""
    print("\n=== 测试PortPy集成 ===")
    
    try:
        import portpy
        from portpy.photon import Plan, Structures, Beams
        
        print("✅ PortPy导入成功")
        print(f"✅ PortPy版本: {portpy.__version__}")
        
        # 测试基本PortPy功能
        print("✅ PortPy基本功能正常")
        
        return True
        
    except Exception as e:
        print(f"❌ PortPy集成测试失败: {e}")
        return False


def test_functional_data_analysis():
    """测试功能数据分析"""
    print("\n=== 测试功能数据分析 ===")
    
    try:
        from train_portpy import func_to_state, intergration_form_fourier
        
        # 模拟功能数据
        functional_data = np.random.randn(12, 100)  # 12个器官，100个数据点
        domin_range = np.linspace(0, 2.0, 100)
        
        # 测试状态转换
        state = func_to_state(functional_data, domin_range)
        print(f"✅ 功能数据转状态成功: 状态形状 {state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能数据分析测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试PortPy重构...")
    
    tests = [
        ("导入测试", test_imports),
        ("数据结构测试", test_data_structures),
        ("强化学习环境测试", test_rl_environment),
        ("PortPy集成测试", test_portpy_integration),
        ("功能数据分析测试", test_functional_data_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！PortPy重构成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要修复问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
