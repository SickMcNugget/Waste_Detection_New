CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --type detr --num-gpus=4 SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0004