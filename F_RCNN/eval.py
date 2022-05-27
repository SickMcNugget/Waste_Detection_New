# Required detectron2 libraries
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Other useful libraries
import os
import argparse

# Personal utilities
from waste_utils import register_waste_dataset
from waste_utils import waste_cfg

parser = argparse.ArgumentParser(description="Evaluate a model from a given directory")
parser.add_argument('path', type=str, help='The path to evaluate')

#Get the path to the model's directory
args = parser.parse_args()

#Make sure the data is visible
register_waste_dataset()

#Set correct configs (mainly for ROI and the test set)
cfg = waste_cfg()

predictor = DefaultPredictor(cfg)
DetectionCheckpointer(predictor.model).load(args.path + "model_final.pth")

evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=args.path)
val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
print(inference_on_dataset(predictor.model, val_loader, evaluator))
