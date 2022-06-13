from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

import torch
from detr.dataset_mapper import DetrDatasetMapper
from detr.config import add_detr_config

#This ensures that DETR is registered properly in detectron2
import detr.detr

from typing import Any, Dict, List, Set
from datetime import datetime
import cv2
import os
import sys
import argparse
import itertools
from abc import ABC, abstractmethod

DATA_PATH=os.path.abspath("../datasets/")

class WasteTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

class WasteTrainerDetr(WasteTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer



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

class BaseSetup(ABC):
    def __init__(self, args):
        self.cfg = get_cfg()
        self.args = args

    def calc_epoch_conversion(self, num_epochs):
        # Since detectron2 uses iterations, a conversion will be required
        dataset_dicts = DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0])
        return (len(dataset_dicts) * num_epochs) // self.cfg.SOLVER.IMS_PER_BATCH

    def setup(self, datasets):
        register_waste_dataset()

        self.get_cfg_defaults()

        self.cfg.DATASETS.TRAIN = (datasets[0],)
        self.cfg.DATASETS.TEST = (datasets[1],)

        # All outputs are written to a time-stamped directory
        self.cfg.OUTPUT_DIR = os.path.join(self.cfg.OUTPUT_DIR, datetime.now().strftime("%d-%b-%Y_%I-%M-%S-%p"))
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        self.update_cfg()

        self.cfg.merge_from_list(self.args.opts)
        self.cfg.freeze()

        setup_logger()

        return self.create_trainer()

    @abstractmethod
    def get_cfg_defaults(self):
        pass

    @abstractmethod
    def update_cfg(self):
        pass

    @abstractmethod
    def create_trainer(self):
        pass

class FasterRCNNSetup(BaseSetup):
    def get_cfg_defaults(self):
        # Get the default config then overrite it with Faster-RCNN's defaults
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

        # Separate this model from others
        self.cfg.OUTPUT_DIR = os.path.join(self.cfg.OUTPUT_DIR, "frcnn")

        # For loading pretrained weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        # For loading in data
        self.cfg.DATALOADER.NUM_WORKERS = 4

        # Model-specific parameters
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    def update_cfg(self):
        # This is the actual batch size of the model
        self.cfg.SOLVER.IMS_PER_BATCH = 4 * self.args.num_gpus
        # Learning rate
        self.cfg.SOLVER.BASE_LR = 5e-3 * self.args.num_gpus
        # Allows stepping down of learning rate at certain steps
        self.cfg.SOLVER.GAMMA = 0.5
        self.cfg.SOLVER.WARMUP_METHOD = "linear"

        # Automatically calculate iterations for 300 epochs
        self.cfg.SOLVER.MAX_ITER = self.calc_epoch_conversion(num_epochs=300)

        # Step down the learning rate at epochs 150, 200 and 250
        self.cfg.SOLVER.STEPS = (
            self.calc_epoch_conversion(num_epochs=150), 
            self.calc_epoch_conversion(num_epochs=200), 
            self.calc_epoch_conversion(num_epochs=250))

        # Warmup rate
        self.cfg.SOLVER.WARMUP_ITERS = self.calc_epoch_conversion(num_epochs=5)
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / self.cfg.SOLVER.WARMUP_ITERS

        # For model saving (5 times per run)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.cfg.SOLVER.MAX_ITER // 5

        # Need a testing period (30 times per run)
        self.cfg.TEST.EVAL_PERIOD = self.cfg.SOLVER.MAX_ITER // 100

    def create_trainer(self):
        return WasteTrainer(self.cfg)

class DetrSetup(BaseSetup):
    def get_cfg_defaults(self):
        add_detr_config(self.cfg)
        self.cfg.merge_from_file("./detr/configs/detr_256_6_6_torchvision.yaml")
        self.cfg.OUTPUT_DIR = os.path.join(self.cfg.OUTPUT_DIR, "detr")

    def update_cfg(self):
        self.cfg.SOLVER.IMS_PER_BATCH = 4 * self.args.num_gpus
        self.cfg.SOLVER.BASE_LR = 1e-5 * self.args.num_gpus
        self.cfg.SOLVER.MAX_ITER = self.calc_epoch_conversion(num_epochs=500)
        self.cfg.SOLVER.STEPS = (self.calc_epoch_conversion(num_epochs=300), self.calc_epoch_conversion(num_epochs=400),)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.cfg.SOLVER.MAX_ITER // 5
        self.cfg.TEST.EVAL_PERIOD = self.calc_epoch_conversion(num_epochs=3) 

    def create_trainer(self):
        return WasteTrainerDetr(self.cfg)


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


