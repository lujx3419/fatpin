# FatPIN
Code for "Automatic radiotherapy treatment planning with deep functional reinforcement learning". For a more detailed description of the method, please refer to the manuscript which will be linked here after acceptance.

## Description

This code is about using reinforcement learning to automate radiation therapy planning. This repository includes both the original MATLAB-based version and a refactored PortPy-based version.

The original implementation was developed based on the interaction between MATLAB R2021a and Python. The reinforcement learning environment is constructed using the MatRad toolkit(https://github.com/e0404/matRad) and the reinforcement learning model is implemented with the PyTorch deep learning framework.

The **PortPy version** (`train_portpy.py`) replaces MATLAB dependencies with PortPy, making the code easier to deploy and maintain.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-fda
- SciPy
- PortPy (for the refactored version)

## Train Model

### Original MATLAB Version
```python
python train.py
```

### PortPy Version (Recommended)
```python
python train_portpy.py
```

