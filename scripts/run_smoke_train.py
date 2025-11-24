#!/usr/bin/env python3
"""Run a short smoke-test of the training loop using qpth solver branch.
This script sets PORTPY_SOLVER=qpth and runs a single-episode, single-patient
short training loop by invoking the project's main training function.
"""
import os, sys
# ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Also add FATPIN directory to sys.path so modules like `portpy_structures`
# and `portpy_functions` can be imported as top-level modules.
# locate the FatPIN directory case-insensitively (some checkouts use 'FatPIN')
fatpin_dir = None
for name in os.listdir(repo_root):
    if name.lower() == 'fatpin' and os.path.isdir(os.path.join(repo_root, name)):
        fatpin_dir = os.path.join(repo_root, name)
        break
if fatpin_dir is None:
    # fallback to conventional name
    fatpin_dir = os.path.join(repo_root, 'FatPIN')
if fatpin_dir not in sys.path:
    sys.path.insert(0, fatpin_dir)

# Enable qpth solver branch
os.environ['PORTPY_SOLVER'] = 'qpth'
# optional mode: 'slack' or 'standard'
os.environ.setdefault('PORTPY_QPTH_MODE', 'standard')

import importlib.util
train_path = os.path.join(fatpin_dir, 'train_portpy.py')
print('Loading training script from', train_path)
spec = importlib.util.spec_from_file_location('train_portpy_smoke', train_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
print('Training module loaded as train_portpy_smoke')

# Short-run overrides (adjust module globals before calling main)
train_mod.Episode = 1
train_mod.max_steps_per_patient = 5
# reduce replay buffer and batch for quickness
train_mod.replay_capacity = 1000
train_mod.batch_size = 8

print('Starting short training run (Episode=1, steps=5) with PORTPY_SOLVER=qpth')
try:
    train_mod.main()
except SystemExit:
    # main may call sys.exit; ignore for smoke test
    pass
print('Smoke test finished')
