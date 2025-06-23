# Dual-domain Adaptive Downsampling and Hybrid Feature Integration for Lightweight Pavement Defect Detection with UAV Images

## Requirements
torch==2.1.1 torchvision=0.16.1 python==3.10.13 numpy==1.26.3 opencv-python==4.9.0 mamba-ssm=1.1.1

## Datasets
> UAVRD:
https://drive.google.com/drive/folders/1pu4FXHn_GbD1Dw__tp6AJGxbiHEF670s?usp=drive_link

> RDD:
https://drive.google.com/drive/folders/1CTUcrvPhoyh6D7vm567wpjFsYDOQ2z1K?usp=drive_link

## Training
> Set MODEL_DIR as the path to the model directory

> Set DATA_DIR as the path to the dataset directory

```python train.py cfg=${MODEL_DIR} data=${DATA_DIR}```



