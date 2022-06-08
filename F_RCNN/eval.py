from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import os
import argparse
from waste_utils import get_cfg_defaults

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model from a given directory")
    parser.add_argument('path', type=str, help='The path to evaluate')

    #Get the path to the model's directory
    args = parser.parse_args()

    #Make sure the data is visible
    cfg = get_cfg_defaults()

    # Ensure the correct datasets are tested
    cfg.DATASETS.TRAIN = ("trash_train_9-COCO_raw",)
    cfg.DATASETS.TEST = ("trash_test_9-COCO_raw",)

    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(args.path + "model_final.pth")

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=args.path)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    main()