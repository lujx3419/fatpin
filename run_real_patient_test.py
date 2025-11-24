#!/usr/bin/env python3
"""
真实患者数据的优化测试脚本
用于验证 PortPy 优化器的维度对应关系
"""
import numpy as np
import os
import traceback
from FatPIN.portpy_structures import load_patient_data
from portpy.photon import Optimization, Structures

def clean_old_lines():
    """清理旧的文件内容"""
    diag_file = '/data/satori_hdd1/lujianxu/reinforce/fatpin/result/diagnostics/dimension_check.txt'
    try:
        with open(diag_file, 'w') as f:
            f.write("维度诊断报告\n")
            f.write("生成时间: 2025-11-02\n\n")
    except:
        pass

def analyze_dimensions(patient_id):
    """分析数据维度并记录诊断信息"""
    print(f"\n=== 开始测试: {patient_id} ===")
    
    diag_file = '/data/satori_hdd1/lujianxu/reinforce/fatpin/result/diagnostics/dimension_check.txt'
    
    try:
        # 1. 加载患者数据
        print("加载患者数据...")
        cst, ct, inf_matrix, plan = load_patient_data(patient_id)
        
        # 2. 获取影响矩阵
        A = getattr(inf_matrix, 'A', None)
        if A is not None:
            print("\n影响矩阵统计:")
            print(f"- shape: {A.shape}")
            if hasattr(A, 'nnz'):
                print(f"- 非零元素: {A.nnz}")
                print(f"- 稀疏度: {A.nnz/(A.shape[0]*A.shape[1])*100:.2f}%")
        
        # 3. 准备PortPy结构
        print("\n创建/获取 PortPy 结构对象...")
        # 优先使用 load_patient_data 返回的 PortPy 对象（兼容性）
        portpy_structs = getattr(cst, 'portpy_structs', None)
        if portpy_structs is None:
            # 如果没有可用的真实 Structures 对象，创建一个轻量级替代以便做诊断
            class _LightStructures:
                def __init__(self):
                    self.structures = {}
                def register_struct(self, name, mask, structure_type='OAR'):
                    self.structures[name] = mask
            portpy_structs = _LightStructures()
            print("警告: 未检测到 cst.portpy_structs，使用轻量级 Structures 作为回退（仅用于诊断）")
        
        # 4. 分类并注册结构（注册到工作副本 work_structs，避免调用不支持的方法）
        target_names = []
        oar_names = []

        # 选择可写的结构容器：如果 portpy_structs 支持 register_struct 则直接使用；
        # 否则创建轻量级副本用于诊断记录
        if hasattr(portpy_structs, 'register_struct'):
            work_structs = portpy_structs
        else:
            class _LightStructures:
                def __init__(self):
                    self.structures = {}
                def register_struct(self, name, mask, structure_type='OAR'):
                    self.structures[name] = mask
            work_structs = _LightStructures()

        print("\n注册结构:")
        for name, struct in cst.structures.items():
            vol_mask = struct.get('volume')
            struct_type = struct.get('type', '').upper()

            if vol_mask is not None:
                # 将体素掩码写入 work_structs（不修改原始 portpy_structs）
                try:
                    work_structs.register_struct(
                        name=name,
                        mask=vol_mask,
                        structure_type='TARGET' if struct_type in ['PTV', 'CTV'] else 'OAR'
                    )
                except Exception:
                    # register_struct 失败则直接写入 structures 字典（防御性）
                    try:
                        work_structs.structures[name] = vol_mask
                    except Exception:
                        pass

                if struct_type in ['PTV', 'CTV']:
                    target_names.append(name)
                else:
                    oar_names.append(name)

                print(f"- {name} ({struct_type}): {np.count_nonzero(vol_mask)} voxels")
        
        # 5. 尝试创建优化器（不是必须，仅用于进一步诊断）
        print("\n尝试创建优化器（可选）...")
        opt = None
        try:
            opt = Optimization(
                plan=plan,
                inf_matrix=inf_matrix,
                structures=portpy_structs
            )
        except TypeError as e:
            print(f"警告: 无法按预期创建 Optimization 对象: {e} (这可能是 PortPy 版本差异)，继续诊断")
        except Exception as e:
            print(f"警告: 创建 Optimization 时出错: {e}")
        
        # 5.5 将 inf_matrix.opt_voxels_dict 的索引回写到 cst.structures（global 和 local opt 索引）
        try:
            opt_vox_dict = getattr(inf_matrix, 'opt_voxels_dict', None)
            if isinstance(opt_vox_dict, dict):
                names_list = opt_vox_dict.get('name', None)
                vox_idx_list = opt_vox_dict.get('voxel_idx', None)

                # 如果 opt_vox_dict 同时包含 name 与 voxel_idx，则尝试按顺序回写
                if names_list is not None and vox_idx_list is not None and len(names_list) == len(vox_idx_list):
                    for nm, g_inds in zip(names_list, vox_idx_list):
                        try:
                            nm_str = str(nm)
                            g_list = [int(x) for x in list(g_inds)] if hasattr(g_inds, '__iter__') else []
                            if nm_str in cst.structures:
                                cst.structures.setdefault(nm_str, {})
                                cst.structures[nm_str]['voxel_idx'] = g_list
                        except Exception:
                            continue

                # 构建 global -> local 映射（opt voxel 全局列表）
                try:
                    global_opt_list = opt_vox_dict.get('voxel_idx', None)
                    # opt_vox_dict['voxel_idx'] may be a list of arrays per structure; if there is a top-level 'voxel_idx' representing optimizer voxel list use it
                    if isinstance(global_opt_list, list) and len(global_opt_list) > 0 and hasattr(global_opt_list[0], '__iter__'):
                        # detect case where voxel_idx is per-structure (list of arrays): build global from union
                        flat = []
                        for arr in global_opt_list:
                            try:
                                flat.extend([int(x) for x in list(arr)])
                            except Exception:
                                continue
                        global_opt_list_flat = sorted(set(flat))
                        global_to_local = {int(g): i for i, g in enumerate(global_opt_list_flat)}
                    else:
                        # try direct list
                        global_opt_list_flat = [int(x) for x in list(global_opt_list)] if global_opt_list is not None else []
                        global_to_local = {int(g): i for i, g in enumerate(global_opt_list_flat)}

                    # 为每个 cst 结构计算 opt_voxel_idx（local indices）
                    for name in list(cst.structures.keys()):
                        g_inds = cst.structures.get(name, {}).get('voxel_idx', [])
                        local_inds = []
                        for gg in g_inds:
                            li = global_to_local.get(int(gg), None)
                            if li is not None:
                                local_inds.append(int(li))
                        local_unique = sorted(set(local_inds)) if len(local_inds) > 0 else []
                        cst.structures.setdefault(name, {})
                        cst.structures[name]['opt_voxel_idx'] = local_unique
                except Exception:
                    # 忽略映射失败
                    pass
        except Exception:
            pass
        # 6. 保存诊断信息
        print("\n保存诊断信息...")
        # 确保目录存在
        os.makedirs(os.path.dirname(diag_file), exist_ok=True)
        with open(diag_file, 'a') as f:
            f.write(f"\n\n=== {patient_id} 优化诊断 ===\n")
            f.write(f"时间: 2025-11-02\n\n")
            
            # 6.1 影响矩阵信息 
            f.write("=== 影响矩阵信息 ===\n")
            if A is not None:
                f.write(f"A.shape = {A.shape}\n")
                if hasattr(A, 'nnz'):
                    f.write(f"非零元素: {A.nnz}\n")
                    f.write(f"稀疏度: {A.nnz/(A.shape[0]*A.shape[1])*100:.2f}%\n")
            
            # 6.2 结构信息
            f.write("\n=== 结构信息 ===\n")
            f.write(f"Target 结构 ({len(target_names)}):\n")
            for name in target_names:
                struct = cst.structures[name]
                vol_mask = struct.get('volume')
                opt_idx = struct.get('opt_voxel_idx', [])
                f.write(f"- {name}:\n")
                f.write(f"  * Volume mask: {np.count_nonzero(vol_mask)} voxels\n")
                f.write(f"  * Opt indices: {len(opt_idx)} indices\n")
                
            f.write(f"\nOAR 结构 ({len(oar_names)}):\n")
            for name in oar_names:
                struct = cst.structures[name]
                vol_mask = struct.get('volume')
                opt_idx = struct.get('opt_voxel_idx', [])
                f.write(f"- {name}:\n")
                f.write(f"  * Volume mask: {np.count_nonzero(vol_mask)} voxels\n")
                f.write(f"  * Opt indices: {len(opt_idx)} indices\n")
            
            # 6.3 优化器内部信息（使用 work_structs 记录）
            f.write("\n=== 优化器状态 ===\n")
            f.write(f"已注册结构数量: {len(work_structs.structures)}\n")
            for name in work_structs.structures:
                mask = work_structs.structures[name]
                try:
                    vox_count = np.count_nonzero(mask)
                except Exception:
                    vox_count = 0
                f.write(f"- {name}: {vox_count} voxels\n")
            
            # 6.4 详细映射信息（opt_voxels_dict, ct_to_dose 映射样例）
            f.write("\n=== 详细映射信息 ===\n")
            # A 行数
            if A is not None:
                f.write(f"A_rows = {A.shape[0]}\n")

            # inf_matrix.opt_voxels_dict
            try:
                opt_vox_dict = getattr(inf_matrix, 'opt_voxels_dict', None)
                f.write(f"inf_matrix.opt_voxels_dict type: {type(opt_vox_dict)}\n")
                if isinstance(opt_vox_dict, dict):
                    f.write(f"opt_voxels_dict keys: {list(opt_vox_dict.keys())}\n")
                    # 写样例条目（前10）
                    for k, v in list(opt_vox_dict.items())[:10]:
                        try:
                            f.write(f"- {k}: type={type(v)}, sample={str(list(v)[:10])}\n")
                        except Exception:
                            f.write(f"- {k}: (unrepresentable)\n")
                # ct_to_dose_voxel_map
                if isinstance(opt_vox_dict, dict) and 'ct_to_dose_voxel_map' in opt_vox_dict:
                    ct_map = opt_vox_dict['ct_to_dose_voxel_map']
                    try:
                        # 写前20个映射样例
                        sample_ct = list(range(min(20, len(ct_map))))
                        mapped_samples = [(i, ct_map[i]) for i in sample_ct]
                        f.write(f"ct_to_dose_voxel_map samples (first {len(mapped_samples)}): {mapped_samples}\n")
                    except Exception:
                        f.write("ct_to_dose_voxel_map: (无法列出样例)\n")
            except Exception as e:
                f.write(f"读取 opt_voxels_dict 时出错: {e}\n")

            # 6.5 每个结构的 global/opt 索引样例（取最多3个结构）
            f.write("\n=== 结构索引样例（最多3个结构） ===\n")
            all_structs = list(cst.structures.keys())
            for name in all_structs[:3]:
                info = cst.structures.get(name, {})
                g_inds = info.get('voxel_idx', [])
                opt_inds = info.get('opt_voxel_idx', [])
                f.write(f"结构: {name}\n")
                f.write(f"- global voxel_idx 长度: {len(g_inds)} 样例: {g_inds[:10]}\n")
                f.write(f"- opt_voxel_idx 长度: {len(opt_inds)} 样例: {opt_inds[:10]}\n")
                
        print(f"\n维度诊断信息已保存到: {diag_file}")
        print("\n诊断完成")
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        traceback.print_exc()
        return False

def main(patient_id='Lung_Patient_2'):
    """主函数"""
    # 1. 清理旧文件
    clean_old_lines()
    
    # 2. 运行维度分析
    success = analyze_dimensions(patient_id)
    
    if success:
        print("\n✓ 分析完成")
    else:
        print("\n✗ 分析失败")

if __name__ == '__main__':
    import sys
    patient_id = sys.argv[1] if len(sys.argv) > 1 else 'Lung_Patient_2'
    main(patient_id)