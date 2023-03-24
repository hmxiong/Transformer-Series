CUDA_VISIBLE_DEVICES=3 python main.py --model_type base \
               --batch_size 2 --no_aux_loss --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/detr-r50-e632da11.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

-m torch.distributed.launch --nproc_per_node=4

detr
- ✅ attention.py 
- ✅ detr.py 
- ✅ matcher.py
- ✅ position_embedding.py
- ✅ DC5 
- segmentation not supported

CUDA_VISIBLE_DEVICES=0 python main.py --model_type deformable \
               --batch_size 2 --num_select 100 --dim_feedforward 1024 --num_queries 300 \ 
               --no_aux_loss --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/r50_deformable_detr-checkpoint.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

CUDA_VISIBLE_DEVICES=4 python main.py --model_type deformable --with_box_refine \
               --batch_size 2 --num_select 100 --dim_feedforward 1024 --num_queries300 \
               --no_aux_loss --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

CUDA_VISIBLE_DEVICES=1 python main.py --model_type deformable --with_box_refine --two_stage \
               --batch_size 2 --num_select 100 --dim_feedforward 1024 --no_aux_loss --num_queries300 \
               --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

deformabel_detr
- ✅ deformable_attn.py 
- ✅ deformabel_detr.py 
- ✅ base_operator
- ✅ two_stage
- ✅ iterative bounding box refinement

CUDA_VISIBLE_DEVICES=0 python main.py --model_type conditional --backbone resnet101 --dilation \
               --batch_size 1 --no_aux_loss --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/ConditionalDETR_r101dc5_epoch50.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

CUDA_VISIBLE_DEVICES=0 python main.py --model_type conditional --dilation \
               --batch_size 1 --no_aux_loss --eval \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/ConditionalDETR_r50dc5_epoch50.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco
when using DC5, batch size must be 1

conditional_detr
- ✅ new attn
- ✅ conditional_detr

CUDA_VISIBLE_DEVICES=3 python main.py --model_type dab \
               --batch_size 1 --no_aux_loss --eval --position_embedding dab \
               --cls_loss_coef 1 --dropout 0.0 --num_select 300 \
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/DAB_R50.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

CUDA_VISIBLE_DEVICES=1 python main.py --model_type deformable --use_dab --with_box_refine\
               --batch_size 2 --no_aux_loss --eval  \
               --cls_loss_coef 1 --dropout 0.0 --num_select 100 --num_queries 300\
               --resume /ssd1/lipengxiang/hmxiong/Transformer-Series/checkpoints/DAB_Deformable_DETR_R50.pth \
               --coco_path /ssd1/lipengxiang/datasets/coco

 在decoder中， 只在第一层进行一次query_pos的scale变化
- ✅ dab_transformer 
- ✅ dab_detr
- ✅ dab_deformable_transformer

dn_detr

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
sudo apt install libnccl2=2.8.4-cuda11.1 libnccl-dev=2.6.4-1+cuda10.0
sudo apt install libnccl2 libnccl-dev
libnccl2=2.6.4-1+cuda10.0 libnccl-dev=2.6.4-1+cuda10.0
nccl-repo-ubuntu1604-2.6.4-ga-cuda10.0_1-1_amd64.deb
nccl-local-repo-ubuntu1804-2.8.4-cuda11.1_1.0-1_amd64.deb