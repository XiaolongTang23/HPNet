# HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention
This repository is the official implementation of [HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention](https://openreview.net/pdf?id=nt9YEGfXf3) published in CVPR 2024.
![AnoverviewofHPNet](assets/HPNet.png)

## Table of Contents
+ [Getting Started](#getting-started)
+ [Training](#training)
+ [Validation](#validation)
+ [Testing](#testing)
+ [Checkpoint & Results](#checkpoint--results)
+ [Reference](#reference)
+ [Acknowledgement](#acknowledgement)

## Getting Started
1\. Clone this repository:
```
git clone https://github.com/XiaolongTang23/HPNet.git
cd HPNet
```

2\. Create a conda environment and install the dependencies:
```
conda create -n HPNet python=3.8
conda activate HPNet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
```
If necessary, you can try combinations of different versions of Python, PyTorch, PyG (PyTorch Geometric), and PyTorch Lightning. For instance, I have successfully tested the project with PyTorch version `1.12.1`, and it worked as well.

3\. Download datasets and install the dependencies:
<details>
<summary><b>Argoverse</b></summary>
<p>

1). Download the [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html#download-link). After downloading and extracting the tar.gz files, organize the dataset directory as follows:

```
/path/to/Argoverse_root/
├── train/
│   └── data/
│       ├── 1.csv
│       ├── 2.csv
│       ├── ...
└── val/
    └── data/
        ├── 1.csv
        ├── 2.csv
        ├── ...
```

2). Install the [Argoverse API](https://github.com/argoverse/argoverse-api).

</p>
</details>

<details>
<summary><b>INTERACTION</b></summary>
<p>

1). Download the [INTERACTION Dataset v1.2](https://interaction-dataset.com/). Here, we only need the data for the multi-agent tracks. After downloading and extracting the zip files, organize the dataset directory as follows:

```
/path/to/INTERACTION_root/
├── maps/
├── test_conditional-multi-agent/
├── test_multi-agent/
├── train/
│   └── DR_CHN_Merging_ZS0_train
│   ├── ...
└── val/
    └── DR_CHN_Merging_ZS0_val
    ├── ...

```

2). Install the map dependency [lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2):
```
pip install lanelet2==1.2.1
```

</p>
</details>


## Training
Data preprocessing may take several hours the first time you run this project. Training on 8 RTX 4090 GPUs, one epoch takes about 30 and 6 minutes for Argoverse and INTERACTION, respectively.
```
# For Argoverse
python HPNet_Argoverse/train.py --root /path/to/Argoverse_root/ --train_batch_size 2 --val_batch_size 2 --devices 8

# For INTERACTION
python HPNet_INTERACTION/train.py --root /path/to/INTERACTION_root/ --train_batch_size 2 --val_batch_size 2 --devices 8
```

## Validation
```
# For Argoverse
python val.py --root /path/to/Argoverse_root/ --val_batch_size 2 --devices 8 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python val.py --root /path/to/INTERACTION_root/ --val_batch_size 2 --devices 8 --ckpt_path /path/to/checkpoint.ckpt
```

## Testing
```
# For Argoverse
python test.py --root /path/to/Argoverse_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python test.py --root /path/to/INTERACTION_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt
```

## Checkpoint & Results
<details>
<summary><b>Argoverse</b></summary>
<p>

We provide a [pre-trained model on Argoverse](https://drive.google.com/file/d/1PqOw3t3-Tf2v6nlqz2bqr0NjYIw_YJwK/view?usp=drive_link), and its results are:
| Split | brier-minFDE | minFDE | MR | minADE |
|----------|:----------:|:----------:|:----------:|:----------:|
| Val | 1.5060 | 0.8708 | 0.0685 | 0.6378 |
| Test | 1.7375 | 1.0986 | 0.1067 | 0.7612 |

</p>
</details>

<!--
<details>
<summary><b>INTERACTION</b></summary>
<p>

Also, we provide a [pre-trained model on INTERACTION](https://drive.google.com/file/d/1PqOw3t3-Tf2v6nlqz2bqr0NjYIw_YJwK/view?usp=drive_link), and its results are:
| Split | minJointFDE | minJointADE |
|----------|:----------:|:----------:|
| Val | 1.5060 | 0.8708 |
| Test | 1.7375 | 1.0986 |

</p>
</details>
-->

## Reference
If you found this repo useful to your research, please consider citing our work:
```
@inproceedings{tang2024hpnet,
  title={HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention},
  author={Xiaolong Tang, Meina Kan, Shiguang Shan, Zhilong Ji, Jinfeng Bai, Xilin CHEN},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Acknowledgement
We sincerely appreciate [Argoverse](https://github.com/argoverse/argoverse-api), [INTERACTION](https://github.com/interaction-dataset/interaction-dataset),[QCNet](https://github.com/ZikangZhou/QCNet) and [HiVT](https://github.com/ZikangZhou/HiVT) for their awesome codebases.
