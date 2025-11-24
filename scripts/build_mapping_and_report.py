#!/usr/bin/env python3
import os, sys, json
# ensure repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from FatPIN.portpy_structures import load_patient_data
from portpy.photon import Optimization

out_dir = 'result/real_opt'
report_path = os.path.join(out_dir, 'report.txt')
map_path = os.path.join(out_dir, 'structure_mapping.json')
dose_path = os.path.join(out_dir, 'dose.npy')

if not os.path.exists(dose_path):
    print('dose.npy missing; run optimization first')
    sys.exit(2)

dose = np.load(dose_path)

cst, ct, inf_matrix, plan = load_patient_data('Lung_Patient_2')
print('Loaded data; inf_matrix shape', getattr(inf_matrix, 'A').shape)

# instantiate Optimization to ensure any lazy opt_voxels are created
try:
    opt = Optimization(plan, inf_matrix=inf_matrix, clinical_criteria=None)
except Exception as e:
    print('Constructing Optimization raised', e)
    opt = None

# Force population of opt_voxels_dict by calling get_opt_voxels_idx for each struct
ps = getattr(cst, 'portpy_structs', None)
struct_names = list(cst.structures.keys())
for nm in struct_names:
    try:
        # call on inf_matrix if available
        if hasattr(inf_matrix, 'get_opt_voxels_idx'):
            _ = inf_matrix.get_opt_voxels_idx(nm)
        # also call on portpy structures if available
        if ps is not None and hasattr(ps, 'get_structure_dose_voxel_indices'):
            _ = ps.get_structure_dose_voxel_indices(nm)
    except Exception:
        pass

# Now attempt to backfill mapping using inf_matrix.opt_voxels_dict
mapping = {}
if hasattr(inf_matrix, 'opt_voxels_dict') and isinstance(inf_matrix.opt_voxels_dict, dict) and 'voxel_idx' in inf_matrix.opt_voxels_dict:
    # robustly convert elements to int (some elements may be array scalars)
    raw_list = list(inf_matrix.opt_voxels_dict['voxel_idx'])
    opt_vox_global_list = []
    for x in raw_list:
        if x is None:
            continue
        try:
            opt_vox_global_list.append(int(x))
        except Exception:
            import numpy as _np
            xa = _np.asarray(x)
            if xa.size == 1:
                try:
                    opt_vox_global_list.append(int(xa.item()))
                except Exception:
                    continue
            else:
                # if element is an array of indices (unexpected), extend
                try:
                    for xi in xa.ravel():
                        opt_vox_global_list.append(int(xi))
                except Exception:
                    continue
    global_to_local = {int(g): i for i, g in enumerate(opt_vox_global_list)}
    for name, info in cst.structures.items():
        # try to get global indices from portpy_structs
        g_inds = None
        if ps is not None:
            try:
                if hasattr(ps, 'get_structure_dose_voxel_indices'):
                    g_inds = ps.get_structure_dose_voxel_indices(name)
                elif hasattr(ps, 'get_structure_voxel_indices'):
                    g_inds = ps.get_structure_voxel_indices(name)
            except Exception:
                g_inds = None
        if g_inds is None:
            g_inds = info.get('voxel_idx')
        if g_inds is None:
            mapping[name] = []
            continue
        local_inds = []
        for gg in g_inds:
            li = global_to_local.get(int(gg), None)
            if li is not None:
                local_inds.append(int(li))
        mapping[name] = sorted(set(local_inds))
else:
    print('inf_matrix.opt_voxels_dict not available; cannot build mapping')

# Save mapping
with open(map_path, 'w') as f:
    json.dump({k: v for k, v in mapping.items()}, f)
print('Saved mapping to', map_path)

# Build report using mapping
lines = ['Structure,NumVoxels,V_1.8Gy_percent,D95_Gy']
for name, vox in mapping.items():
    vox = np.array(vox, dtype=int)
    if vox.size == 0:
        lines.append(f'{name},0,0.0,0.0')
        continue
    vox = vox[(vox >= 0) & (vox < dose.size)]
    if vox.size == 0:
        lines.append(f'{name},0,0.0,0.0')
        continue
    sd = dose[vox]
    v18 = 100.0 * float(np.sum(sd >= 1.8) / sd.size)
    d95 = float(np.percentile(sd, 5))
    lines.append(f'{name},{sd.size},{v18:.3f},{d95:.4f}')

with open(report_path, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote report to', report_path)
print('\n'.join(lines))
