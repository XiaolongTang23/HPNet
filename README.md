# HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention
This repository is the official implementation of HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention published in CVPR 2024.
![AnoverviewofHPNet](assets/HPNet.png)

## Table of Contents
+ Getting Started
+ Training
+ Validation
+ Testing
+ Pre-trained Model
+ Citation
+ Licence

## Getting Started
1. Clone this repository:
```bash
git clone https://github.com/XiaolongTang23/HPNet.git
cd HPNet
```
2. Create a conda environment and install the dependencies:
```bash
conda create -n HPNet python=3.8
conda activate HPNet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
```
If your hardware cannot support the specified environment configuration, you are encouraged to try combinations of different versions of Python, PyTorch, PyG (PyTorch Geometric), and PyTorch Lightning. In most cases, alternative version combinations should work adequately. For instance, I have successfully tested the project with PyTorch version `1.12.1`, and it worked as well.
 

