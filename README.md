# Transformer-Series
## Introduction
This project is mainly used to complete the paper reproduction and precision alignment, it will be updated continuously. At present, it mainly supports the DETR series of object detection algorithms.The code continues the simple style of Facebook AI Research, which is mainly for the convenience of subsequent maintenance and learning.

![DETR](pics/DETR.png)

## Environment
- CUDA > 11.1  torch >= 1.10.1  torchvision >= 0.10.1 
- numpy = 1.23.5 pycocotools = 2.0  scipy = 1.7.1

## What's New
- Support **DAB-DETR** and **DAN-Deformable-DETR** .



## Model List
<details open>
<summary> Supported methods </summary>

- ✅ [DETR](./result_record/DETR_Precision_alignment_record.md)
- ✅ [Deformable-DETR](./result_record/Deformable-DETR_Precision_alignment.md)
- ✅ [Conditional-DETR](./result_record/Conditional_DETR_Precision_alignment.md)
- ✅ [DAB-DETR](./result_record/DAB_DETR_Precision_alignment.md)
- ✅ [DAB-Deformable-DETR](./result_record/DAB_DETR_Precision_alignment.md)
-  [DN-DETR](comming soon)
-  [DINO]
</details>


## Usage - Object detection
First, clone the repository locally:
```bash
git clone 
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```bash
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```bash
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Install other requirements:
```bash
pip install -r requirements.txt
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```
Compiling CUDA operators
```bash
cd ./model/basic_operator
python setup.py build install
# unit test (should see all checking is True)
python test.py
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```bash
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

## Evaluation
To evaluate DETR R50 on COCO val5k with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/checkpoints --coco_path /path/to/coco
```