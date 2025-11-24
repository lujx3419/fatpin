#!/usr/bin/env python3
import os, time, pickle, traceback, sys
import numpy as np

# Ensure repository root is on sys.path so `FatPIN` package can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from FatPIN.portpy_structures import load_patient_data, PortPyOptimizer

out_dir = 'result/real_opt'
os.makedirs(out_dir, exist_ok=True)
status_path = os.path.join(out_dir, 'status.txt')
summary_path = os.path.join(out_dir, 'summary.txt')

try:
    start_all = time.time()
    print('Loading patient data...')
    cst, ct, inf_matrix, plan = load_patient_data('Lung_Patient_2')
    print('Loaded data: plan type =', type(plan))

    po = PortPyOptimizer()
    print('Starting optimization (this may take a while)...')
    t0 = time.time()
    res, info = po.optimize(inf_matrix, cst, plan)
    t1 = time.time()
    took = t1 - t0
    print(f'Optimization finished in {took:.1f}s')

    # Save results
    if res is None:
        with open(status_path, 'w') as f:
            f.write('failed')
        print('No result returned (res is None)')
    else:
        # Save sol dict if present
        sol = res.get('sol') if isinstance(res, dict) else None
        try:
            if sol is not None:
                with open(os.path.join(out_dir, 'sol.pkl'), 'wb') as f:
                    pickle.dump(sol, f)
                print('Saved sol.pkl')
        except Exception as e:
            print('Warning: failed to pickle sol:', e)

        # Save fluence and dose if available
        try:
            flu = res.get('fluence') if isinstance(res, dict) else None
            dose = res.get('dose') if isinstance(res, dict) else None
            if flu is not None:
                np.save(os.path.join(out_dir, 'fluence.npy'), np.asarray(flu))
                print('Saved fluence.npy')
            if dose is not None:
                np.save(os.path.join(out_dir, 'dose.npy'), np.asarray(dose))
                print('Saved dose.npy')
        except Exception as e:
            print('Warning: failed saving arrays:', e)

        with open(status_path, 'w') as f:
            f.write('success')

        # Summary
        with open(summary_path, 'w') as f:
            f.write('Optimization summary\n')
            f.write(f'time_sec: {took:.2f}\n')
            f.write(f'info: {repr(info)}\n')
            f.write(f'res keys: {list(res.keys()) if isinstance(res, dict) else None}\n')
        print('Wrote summary.txt')

    end_all = time.time()
    print('Total wall time:', end_all - start_all)
except Exception as e:
    traceback.print_exc()
    with open(status_path, 'w') as f:
        f.write('failed')
    with open(summary_path, 'w') as f:
        f.write('Exception during run:\n')
        f.write(str(e) + '\n')
    print('Optimization run failed; see run.log and summary.txt')
