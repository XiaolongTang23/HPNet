# HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention
This repository is the official implementation of HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention published in CVPR 2024.
![AnoverviewofHPNet](assets/HPNet.png)

## Table of Contents
+ [Getting Started](#getting-started)
+ [Training](#training)
+ [Validation](#validation)
+ [Testing](#testing)
+ [Checkpoint & Results](#checkpoint--results)
+ [Reference](#reference)

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
If necessary, you can try combinations of different versions of Python, PyTorch, PyG (PyTorch Geometric), and PyTorch Lightning. For instance, I have successfully tested the project with PyTorch version `1.12.1`, and it worked as well.

3. Download [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html#download-link), extract the Dataset Files, and [install Argoverse API](https://github.com/argoverse/argoverse-api).

## Training
For the initial training, data preprocessing may take several hours. Training on 8 RTX 4090 GPUs, one epoch takes about 30 minutes.
```bash
python train.py --root /path/to/dataset_root/ --train_batch_size 2 --val_batch_size 2 --devices 8
```

## Validation
```bash
python val.py --root path/to/dataset_root --val_batch_size 2 --devices 8 --ckpt_path path/to/checkpoint.ckpt 
```

## Testing
```bash
python test.py --root path/to/dataset_root --test_batch_size 2 --devices 1 --ckpt_path path/to/checkpoint.ckpt 
```

## Checkpoint & Results
We provide a [pre-trained model](https://drive.google.com/file/d/1PqOw3t3-Tf2v6nlqz2bqr0NjYIw_YJwK/view?usp=drive_link), and its result on Argoverse is:
| Split | brier-minFDE | minnFDE | MR | minADE |
|----------|----------|----------|----------|----------|
| Val | 1.5060 | 0.8708 | 0.0685 | 0.6378 |
| Test | 1.7375 | 1.0986 | 0.1067 | 0.7612 |

## Reference
If you found this repo useful to your research, please consider citing our work:
```bash
@inproceedings{tang2024hpnet,
  title={HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention},
  author={Xiaolong Tang, Meina Kan, Shiguang Shan, Zhilong Ji, Jinfeng Bai, Xilin CHEN},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```


