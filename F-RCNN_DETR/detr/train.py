from detectron2.engine import launch
from detectron2.utils.logger import setup_logger

import os
import sys

# See into the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from waste_utils import WasteTrainerDetr, get_cfg_defaults_detr, default_argument_parser, update_cfg_detr

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_defaults_detr()

    # Datasets
    cfg.DATASETS.TRAIN = ("trash_train_10-COCO_raw",)
    cfg.DATASETS.TEST = ("trash_test_10-COCO_raw",)
    cfg.MODEL.DETR.NUM_CLASSES = 10
    
    # Add specific config options for DETR
    update_cfg_detr(cfg, args)

    # Allows changes to be made from the command line
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)

    # Make sure detectron2 is ready to log metrics
    setup_logger()
    trainer = WasteTrainerDetr(cfg)
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