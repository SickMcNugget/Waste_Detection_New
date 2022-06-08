from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import torch

from datetime import datetime
import random
import cv2
import os
import sys
import argparse

DATA_PATH=os.path.abspath("../datasets/")

class WasteTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

class WasteVisualizer(object):
    def __init__(self, cfg):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "NA"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE
        self.predictor = DefaultPredictor(cfg)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame))

def register_waste_dataset():
    """Automatically finds and registers datasets"""
    # Get all datasets in the datasets folder
    datasets = []

    for item in os.listdir(DATA_PATH):
        # Skip all files
        if not os.path.isdir(os.path.join(DATA_PATH, item)):
            continue

        # The main type will be TACO_TN_UAV
        if "TACO_TN_UAV" in item:
            # Easier to remember than the full name
            prefix = "trash_"
            # Get everything that isnt TACO_TN_UAV into a string
            suffix = item.replace("TACO_TN_UAV", "")
        else:
            # All other datasets will be referred to with their full name
            prefix = item
            suffix = ""

        cur_path = os.path.join(DATA_PATH, item)
        for subdir in os.listdir(cur_path):
            if subdir == "train" or subdir == "valid" or subdir == "test":
                final_path = os.path.join(cur_path, subdir)
                register_coco_instances(f"{prefix}{subdir}{suffix}", 
                    {}, 
                    f"{final_path}/_annotations.coco.json", 
                    final_path)
                datasets.append(f"{prefix}{subdir}{suffix}")

def get_cfg_defaults():
    """Load the model that will always be used in this project.
    After loading the model, override default parameters that will not change
    """

    # Make sure detectron knows where the datasets are
    register_waste_dataset()

    # Get the default config then overrite it with Faster-RCNN's defaults
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # All outputs are written to a time stamped directory
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datetime.now().strftime("%d-%b-%Y_%I-%M-%S-%p"))

    # For loading pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.DATALOADER.NUM_WORKERS = 8

    # Model-specific parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return cfg
 
def calc_epoch_conversion(cfg, num_epochs):
    # Since detectron2 uses iterations, a conversion will be required
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    return (len(dataset_dicts) * num_epochs) // cfg.SOLVER.IMS_PER_BATCH

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # parser.add_argument("data_path", default="", metavar="DATA_ROOT", help="Path to the folder containing COCO datasets")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser