CUDA_VISIBLE_DEVICES=0,1,3,4 python train.py --type frcnn --num-gpus=4 SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.02