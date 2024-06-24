# HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention
[arXiv](https://arxiv.org/pdf/2404.06351.pdf) | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_HPNet_Dynamic_Trajectory_Forecasting_with_Historical_Prediction_Attention_CVPR_2024_paper.pdf) | [poster](assets/poster_2024_cvpr_hpnet.png)\
This repository is the official implementation of HPNet: Dynamic Trajectory Forecasting with Historical Prediction Attention published in CVPR 2024.
![AnoverviewofHPNet](assets/HPNet.png)

## Table of Contents
+ [Setup](#setup)
+ [Datasets](#datasets)
+ [Training](#training)
+ [Validation](#validation)
+ [Testing](#testing)
+ [Pre-trained Models & Results](#pre-trained-models--results)
+ [Acknowledgements](#acknowledgements)
+ [Citation](#citation)

## Setup
Clone the repository and set up the environment:
```
git clone https://github.com/XiaolongTang23/HPNet.git
cd HPNet
conda create -n HPNet python=3.8
conda activate HPNet
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric==2.3.1
conda install pytorch-lightning==2.0.3
```
*Note:* For compatibility, you may experiment with different versions, e.g., PyTorch 1.12.1 has been confirmed to work.

## Datasets

<details>
<summary><b>Argoverse</b></summary>
<p>

1. Download the [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html#download-link). After downloading and extracting the tar.gz files, organize the dataset directory as follows:

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

2. Install the [Argoverse API](https://github.com/argoverse/argoverse-api).

</p>
</details>

<details>
<summary><b>INTERACTION</b></summary>
<p>

1. Download the [INTERACTION Dataset v1.2](https://interaction-dataset.com/). Here, we only need the data for the multi-agent tracks. After downloading and extracting the zip files, organize the dataset directory as follows:

```
/path/to/INTERACTION_root/
├── maps/
├── test_conditional-multi-agent/
├── test_multi-agent/
├── train/
│   ├── DR_CHN_Merging_ZS0_train
│   ├── ...
└── val/
    ├── DR_CHN_Merging_ZS0_val
    ├── ...

```

2. Install the map dependency [lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2):
```
pip install lanelet2==1.2.1
```

</p>
</details>


## Training
Data preprocessing may take several hours the first time you run this project. Training on 8 RTX 4090 GPUs, one epoch takes about 30 and 6 minutes for Argoverse and INTERACTION, respectively.
```
# For Argoverse
python HPNet-Argoverse/train.py --root /path/to/Argoverse_root/ --train_batch_size 2 --val_batch_size 2 --devices 8

# For INTERACTION
python HPNet-INTERACTION/train.py --root /path/to/INTERACTION_root/ --train_batch_size 2 --val_batch_size 2 --devices 8
```

## Validation
```
# For Argoverse
python HPNet-Argoverse/val.py --root /path/to/Argoverse_root/ --val_batch_size 2 --devices 8 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python HPNet-INTERACTION/val.py --root /path/to/INTERACTION_root/ --val_batch_size 2 --devices 8 --ckpt_path /path/to/checkpoint.ckpt
```

## Testing
```
# For Argoverse
python HPNet-Argoverse/test.py --root /path/to/Argoverse_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt

# For INTERACTION
python HPNet-INTERACTION/test.py --root /path/to/INTERACTION_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt
```

## Pre-trained Models & Results

### Argoverse
- **Pre-trained model:** [Download here](https://drive.google.com/file/d/1PqOw3t3-Tf2v6nlqz2bqr0NjYIw_YJwK/view?usp=drive_link)
- **Performance Metrics:**

| Split | brier-minFDE | minFDE | MR | minADE |
|-------|:------------:|:------:|:--:|:------:|
| Val   | 1.5060       | 0.8708 | 0.0685 | 0.6378 |
| Test  | 1.7375       | 1.0986 | 0.1067 | 0.7612 |

### INTERACTION
- **Pre-trained model:** [Download here](https://drive.google.com/file/d/1wj6Wg2-eta4pVFxHARsaVCyisk2Fr-qM/view?usp=sharing)
- **Performance Metrics:**

| Split | minJointFDE | minJointADE |
|-------|:-----------:|:-----------:|
| Val   | 0.5577      | 0.1739      |
| Test  | 0.8231      | 0.2548      |


## Acknowledgements
We sincerely appreciate [Argoverse](https://github.com/argoverse/argoverse-api), [INTERACTION](https://github.com/interaction-dataset/interaction-dataset),[QCNet](https://github.com/ZikangZhou/QCNet) and [HiVT](https://github.com/ZikangZhou/HiVT) for their awesome codebases.


## Citation

If HPNet has been helpful in your research, please consider citing our work:

```@inproceedings{tang2024hpnet,
  title={Hpnet: Dynamic trajectory forecasting with historical prediction attention},
  author={Tang, Xiaolong and Kan, Meina and Shan, Shiguang and Ji, Zhilong and Bai, Jinfeng and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15261--15270},
  year={2024}
}
```
