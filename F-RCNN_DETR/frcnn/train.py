from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
import os
import sys

# See into the parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from waste_utils import WasteTrainer, get_cfg_defaults_frcnn, default_argument_parser, update_cfg_frcnn

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # Load some default config parameters
    cfg = get_cfg_defaults_frcnn()

    # The registered datasets to use
    cfg.DATASETS.TRAIN = ("trash_train_10-COCO_raw",)
    cfg.DATASETS.TEST = ("trash_test_10-COCO_raw",)

    # Add specific config options for Faster R-CNN
    update_cfg_frcnn(cfg, args)

    # Allows changes to be made from the command line
    cfg.merge_from_list(args.opts)

    # Make sure the config is now frozen as-is
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)

    # Make sure detectron2 is ready to log metrics
    setup_logger()
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