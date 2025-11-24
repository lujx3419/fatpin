#!/usr/bin/env python3
import os, sys, pickle
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from FatPIN.portpy_structures import load_patient_data, PortPyOptimizer
import portpy.photon as pp

out_dir = 'result/real_opt'
fluence_path = os.path.join(out_dir, 'fluence.npy')
dose_path = os.path.join(out_dir, 'dose.npy')
map_path = os.path.join(out_dir, 'cst_with_mapping.pkl')
report_path = os.path.join(out_dir, 'report.txt')

if not os.path.exists(fluence_path):
    print('fluence.npy missing; cannot fake solve')
    sys.exit(2)

fluence = np.load(fluence_path)

# load data
cst, ct, inf_matrix, plan = load_patient_data('Lung_Patient_2')
print('Loaded; creating optimizer')
po = PortPyOptimizer()

# monkeypatch Optimization.solve to return saved fluence
OptClass = pp.optimization.Optimization
orig_solve = getattr(OptClass, 'solve', None)

def fake_solve(self, *args, **kwargs):
    print('fake_solve called')
    return {'optimal_intensity': fluence, 'inf_matrix': self.inf_matrix}

OptClass.solve = fake_solve

try:
    res, info = po.optimize(inf_matrix, cst, plan)
    # After optimize, try to extract mapping from cst
    mapping = {name: info2.get('opt_voxel_idx') or info2.get('voxel_idx') or [] for name, info2 in cst.structures.items()}
    # save mapping only (safe to serialize)
    import json
    with open(map_path, 'w') as f:
        json.dump(mapping, f)
    print('Saved mapping to', map_path)
    # generate report using dose.npy
    if os.path.exists(dose_path):
        dose = np.load(dose_path)
        lines = ['Structure,NumVoxels,V_1.8Gy_percent,D95_Gy']
        for name, vox in mapping.items():
            vox = np.array(vox, dtype=int) if vox is not None else np.array([], dtype=int)
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
    else:
        print('dose.npy missing; cannot compute report')
finally:
    # restore solve
    if orig_solve is not None:
        OptClass.solve = orig_solve
