import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# Initialise logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# required libraries
import numpy as np
import cv2, random, os

# utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Personal utilities
from waste_utils import show_dataset

# Create a config and predictor to run a prediction
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("smth"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("smth")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(image)

#training
from detectron2.data.datasets import register_coco_instances
register_coco_instances("trash_train", 
    {}, 
    "/home/joren/Documents/Trash_Dataset_Full/train/_annotations.coco.json",
    "/home/joren/Documents/Trash_Dataset_Full/train/")
register_coco_instances("trash_valid",
    {},
    "/home/joren/Documents/Trash_Dataset_Full/valid/_annotations.coco.json",
    "/home/joren/Documents/Trash_Dataset_Full/valid/")

# test that datasets registered correctly
# show_dataset("trash_train", 3)

# Training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("trash_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()