# Transformer-Series
## Introduction
This project is mainly used to complete the paper reproduction and precision alignment, it will be updated continuously. At present, it mainly supports the DETR series of object detection algorithms.The code continues the simple style of Facebook AI Research, which is mainly for the convenience of subsequent maintenance and learning.

![DETR](pics/DETR.png)

## Environment
- CUDA > 11.1  torch >= 1.10.1  torchvision >= 0.10.1 
- numpy = 1.23.5 pycocotools = 2.0  scipy = 1.7.1

## What's New
- Support **DETR** whole process of training and evaluation on COCO.
- Support **Deformable-DETR** only evaluation on COCO.


## Model List
<details open>
<summary> Supported methods </summary>

- ✅ [DETR](./result_record/DETR_Precision_alignment_record.md)
- ✅ [Deformable-DETR](./result_record/Deformable-DETR_Precision_alignment.md)
-  [Conditional-DETR] (comming soon)
-  [DAB-DETR] 
-  [DAB-Deformable-DETR]

</details>

# Usage - Object detection
First, clone the repository locally:
```
git clone 
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Install other requirements:
```
pip install -r requirements.txt
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
At present, only the basic version of DETR training is supported.

## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/checkpoints --coco_path /path/to/coco
```