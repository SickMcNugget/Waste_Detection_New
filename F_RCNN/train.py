import torch
from detectron2.engine import DefaultTrainer, launch
from detectron2.utils.logger import setup_logger
import os
from waste_utils import get_cfg_defaults, default_argument_parser, calc_epoch_conversion
import argparse
import sys

def main(args):
    
    # Load some default config parameters
    cfg = get_cfg_defaults(args)

    # The registered datasets to use
    cfg.DATASETS.TRAIN = ("trash_train_raw",)
    cfg.DATASETS.TEST = ("trash_test_raw",)

    # This is the actual batch size of the model
    cfg.SOLVER.IMS_PER_BATCH = 8 * args.num_gpus
    
    # Automatically calculate iterations for 300 epochs
    cfg.SOLVER.MAX_ITER = calc_epoch_conversion(cfg, num_epochs=300)

    # Learning rate
    cfg.SOLVER.BASE_LR = 5e-3 * args.num_gpus

    # Allows stepping down of learning rate at certain steps
    cfg.SOLVER.GAMMA = 0.5
    # Step down the learning rate at epochs 150, 200 and 250
    cfg.SOLVER.STEPS = (
        calc_epoch_conversion(cfg, num_epochs=150), 
        calc_epoch_conversion(cfg, num_epochs=200), 
        calc_epoch_conversion(cfg, num_epochs=250))

    # Warmup rate
    cfg.SOLVER.WARMUP_ITERS = calc_epoch_conversion(cfg, num_epochs=5)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.WARMUP_METHOD = "linear"

    # For model saving (5 times per run)
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 5

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Make sure detectron2 is ready to log metrics
    setup_logger()
    # Weights and biases will use the detectron2 logger to upload data
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )