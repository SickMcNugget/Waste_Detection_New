import torch
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
import os
from waste_utils import WasteTrainer, get_cfg_defaults, default_argument_parser, update_cfg
import argparse
import sys

def main(args):
    
    # Load some default config parameters
    cfg = get_cfg_defaults()

    # Add specific config options for Faster R-CNN
    update_cfg_fcnn(cfg, args)

    # The registered datasets to use
    cfg.DATASETS.TRAIN = ("trash_train_9-COCO_raw",)
    cfg.DATASETS.TEST = ("trash_test_9-COCO_raw",)

    # Allows changes to be made from the command line
    cfg.merge_from_list(args.opts)

    # Make sure the config is now frozen as-is
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Make sure detectron2 is ready to log metrics
    setup_logger()
    # Weights and biases will use the detectron2 logger to upload data
    trainer = WasteTrainer(cfg)
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