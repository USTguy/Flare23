# FLARE23 Solution

This repository contains our solution "Iterative Semi-Supervised Learning for Abdominal Organs and Tumor Segmentation" for the FLARE23 challenge, based on nnU-Netv2.

## 🔍 Overview

Our approach is based on interative SSL, which employs a multi-stage pseudo-labeling method to tackle the issue of partial labels for organs and tumors in the FLARE23 dataset. For more details, see the pipeline diagram below:

<img src="./frame.png" alt="Pipeline" width="600"/>

## ⚙️ Environment Setup

To set up the environment, follow these steps:

```
conda create -n FLARE23
conda activate FLARE23
```
Then make sure to install PyTorch 2 compatible with your CUDA version.
```
pip install -e .
```

## ⬇️ Download Checkpoints

Download the checkpoints and pseudo labels from [BaiduNetDisk](https://pan.baidu.com/s/1yFmK8WZifE-EhFQTcfsYtA?pwd=1111 ). Code：1111

## 🚀 Inference

1. Place your input images in the `./inputs` directory.
2. Run the prediction script:

```
sh predict.sh
```

This will generate the output in the `./outputs` directory.


# Flare23
Our solution for MICCAI Flare23

Follow the instructions of nn-UNet to learn a teacher model with labeled data. The code to select labeled data is placed in the 'process' folder. Our generated jsons are in the 'json' folder.

Generate pseudo labels using nn-unet for unlabeled data.

Our checkpoints and generate pseudo labels are available at Baiduyun. Link：https://pan.baidu.com/s/1yFmK8WZifE-EhFQTcfsYtA?pwd=1111 
Code：1111

More details are coming!
