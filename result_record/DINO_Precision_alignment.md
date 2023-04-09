# Evaluation Results：
## Environment
- Tesla V100
- CUDA 11.1
- CUDNN 8.0
- python = 3.8
- torch = 1.10.1
- torchvision = 0.10.1
- scipy = 1.7.1
- numpy = 1.23.5
- pycocotools = 2.0
## DINO-R50-Scale4
Command:
```bash
python main.py --model_type dino --use_dn\
               --batch_size 2 --no_aux_loss --eval --position_embedding dab \
               --cls_loss_coef 1.0 --dn_box_noise_scale 1.0 \
               --cls_loss_coef 1 --dropout 0.0 --num_select 300  --num_queries 900\
               --dec_pred_bbox_embed_share --dec_pred_class_embed_share \
               --embed_init_tgt --match_unstable_error --resume path/to/checkpoints \
               --transformer_activation relu\
               --resume path/to/checkpoints \
               --coco_path path/to/coco
```
COCO detection val5k evaluation results:
```bash
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.508
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.690
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.540
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.872
```
### notice
