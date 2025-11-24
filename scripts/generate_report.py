#!/usr/bin/env python3
import os, sys, numpy as np
# ensure repo root on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from FatPIN.portpy_structures import load_patient_data

out_dir = 'result/real_opt'
report_path = os.path.join(out_dir, 'report.txt')

dose_path = os.path.join(out_dir, 'dose.npy')
if not os.path.exists(dose_path):
    print('dose.npy not found at', dose_path)
    sys.exit(2)

dose = np.load(dose_path)
print('Loaded dose array shape:', dose.shape)

# load cst to get structure voxel indices
cst, ct, inf_matrix, plan = load_patient_data('Lung_Patient_2')

lines = []
lines.append('Structure,NumVoxels,V_1.8Gy_percent,D95_Gy')
for name, info in cst.structures.items():
    vox_idx = info.get('opt_voxel_idx') or info.get('voxel_idx')
    # If mapping missing, try to backfill from inf_matrix.opt_voxels_dict
    if (vox_idx is None or len(vox_idx) == 0) and hasattr(inf_matrix, 'opt_voxels_dict'):
        try:
            od = inf_matrix.opt_voxels_dict
            if isinstance(od, dict) and 'voxel_idx' in od:
                opt_vox_global = [int(x) for x in list(od['voxel_idx']) if x is not None]
                global_to_local = {int(g): i for i, g in enumerate(opt_vox_global)}
                # try to get per-structure global indices from portpy_structs if available
                g_inds = None
                ps = getattr(cst, 'portpy_structs', None)
                if ps is not None:
                    try:
                        if hasattr(ps, 'get_structure_dose_voxel_indices'):
                            g_inds = ps.get_structure_dose_voxel_indices(name)
                        elif hasattr(ps, 'get_structure_voxel_indices'):
                            g_inds = ps.get_structure_voxel_indices(name)
                    except Exception:
                        g_inds = None
                # fallback to any existing 'voxel_idx' field
                if g_inds is None:
                    g_inds = info.get('voxel_idx')
                if g_inds is not None:
                    local_inds = []
                    for gg in g_inds:
                        li = global_to_local.get(int(gg), None)
                        if li is not None:
                            local_inds.append(int(li))
                    vox_idx = sorted(set(local_inds))
        except Exception:
            pass
    if vox_idx is None:
        vox_idx = []
        try:
            if hasattr(cst, 'portpy_structs') and cst.portpy_structs is not None:
                # try dose voxel mapping first
                if hasattr(cst.portpy_structs, 'get_structure_dose_voxel_indices'):
                    inds = cst.portpy_structs.get_structure_dose_voxel_indices(name)
                else:
                    inds = None
                if inds is None and hasattr(cst.portpy_structs, 'get_structure_voxel_indices'):
                    inds = cst.portpy_structs.get_structure_voxel_indices(name)
                if inds is not None:
                    vox_idx = list(map(int, inds))
        except Exception:
            vox_idx = []
    vox_idx = np.array(vox_idx, dtype=int)
    if vox_idx.size == 0:
        lines.append(f'{name},0,0.0,0.0')
        continue
    vox_idx = vox_idx[(vox_idx >= 0) & (vox_idx < dose.size)]
    if vox_idx.size == 0:
        lines.append(f'{name},0,0.0,0.0')
        continue
    struct_doses = dose[vox_idx]
    v18 = 100.0 * float(np.sum(struct_doses >= 1.8) / struct_doses.size)
    d95 = float(np.percentile(struct_doses, 5))
    lines.append(f'{name},{struct_doses.size},{v18:.3f},{d95:.4f}')

with open(report_path, 'w') as f:
    f.write('\n'.join(lines))

print('Wrote report to', report_path)
print('\n'.join(lines))
