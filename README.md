# MobileNetv2 on CIFAR-10
This repository implements standard deep learning model compression techniques (Pruning + Quantization) to build more compact models for edge devices

# Repository Structure 
1. models/ - Includes various model checkpoints for easy reproducibility
2. scripts/ - Includes pythin scripts to train/prune/quantize the MobileNetV2 model on CIFAR-10 dataset
3. notebooks/ - Jupyter notebooks with equivalent implementation
4. utils/ - Helper scripts for compression analysis

# Development Enviornment Setup
Follow the below steps to setup your local system to run the model scripts

## Setting up Conda(Optional)
Refer to the official documentation for more details - https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html

For installing pre-requisities, use 
```pip install -r requirements.txt```

# System Configuration
```
Package                   Installed Version
---------------------------------------------
torch                     2.9.0
numpy                     2.3.4
Pillow                    12.0.0
matplotlib                3.10.7
torchvision               0.24.0
Python                    3.13.5
```
