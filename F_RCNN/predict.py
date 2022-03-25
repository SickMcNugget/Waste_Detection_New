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

# utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Create a config and predictor to run a prediction
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("smth"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("smth")
predictor = DefaultPredictor(cfg)
outputs = predictor(image)
