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
from datetime import datetime
import os

# utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Personal utilities
from waste_utils import show_dataset
from waste_utils import register_waste_dataset

# Create a config and predictor to run a prediction
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("smth"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("smth")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(image)

#training
register_waste_dataset()

# test that datasets registered correctly
# show_dataset("trash_train", 3)

# Training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datetime.now().strftime("%d-%b-%Y_%-I:%M:%S_%p"))

cfg.DATASETS.TRAIN = ("trash_train",)
cfg.DATASETS.TEST = ("trash_test",)

cfg.DATALOADER.NUM_WORKERS = 8

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo


cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.MAX_ITER = 25000

#Final lr is calculated by lrf = base_lr * (gamma ^ (step threshold))
#                       i.e [it 0]     lrf = 5e-3 * (0.1 ^ 0)
#                           [it 15000] lrf = 5e-3 * (0.1 ^ 1)
#                           [it 20000] lrf = 5e-3 * (0.1 ^ 2)
cfg.SOLVER.BASE_LR = 5e-3  # pick a good LR
cfg.SOLVER.GAMMA=0.1
cfg.SOLVER.STEPS = (15000,20000,)
cfg.SOLVER.WARMUP_FACTOR = 0.5 / 2000
cfg.SOLVER.WARMUP_ITERS = 2000
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.SOLVER.REFERENCE_WORLD_SIZE = 1

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 192
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
