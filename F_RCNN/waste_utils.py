from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import torch

from datetime import datetime
import random
import cv2
import os

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
    register_coco_instances("trash_train", 
        {}, 
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/train/_annotations.coco.json",
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/train")
    register_coco_instances("trash_valid",
        {},
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/valid/_annotations.coco.json",
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/valid/")
    register_coco_instances("trash_test",
        {},
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/test/_annotations.coco.json",
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO/test/")

    register_coco_instances("trash_train_raw", 
        {}, 
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO_raw/train/_annotations.coco.json",
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO_raw/train")
    register_coco_instances("trash_test_raw",
        {},
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO_raw/test/_annotations.coco.json",
        "/home/joren/Documents/Waste_Detection_New/datasets/TACO_TN_UAV_10-COCO_raw/test/")

def waste_cfg(yaml="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml))

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datetime.now().strftime("%d-%b-%Y_%-I:%M:%S_%p"))

    cfg.DATASETS.TRAIN = ("trash_train_raw",)
    cfg.DATASETS.TEST = ("trash_test_raw",)

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml)  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.MAX_ITER = 35000

    #Final lr is calculated by lrf = base_lr * (gamma ^ (step threshold))
    #                       i.e [it 0]     lrf = 2e-3 * (0.5 ^ 0)
    #                           [it 15000] lrf = 2e-3 * (0.5 ^ 1)
    #                           [it 20000] lrf = 2e-3 * (0.5 ^ 2)
    cfg.SOLVER.BASE_LR = 4e-3  # pick a good LR
    cfg.SOLVER.GAMMA=0.3
    cfg.SOLVER.STEPS = (15000,25000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3000
    cfg.SOLVER.WARMUP_ITERS = 3000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    return cfg
 
def show_dataset(name, num=3):
    dataset_dicts = DatasetCatalog.get(name)
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

def predict(name, predictor, num=3):
    dataset_dicts = DatasetCatalog.get(name)
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], scale=0.5)
        for box in outputs["instances"].pred_boxes.to('cpu'):
            v.draw_box(box)
            v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
        v = v.get_output()
        img = v.get_image()[:, :, ::-1]
        cv2.imshow(d["file_name"], img)
        cv2.waitKey(0)

def predict2(name, predictor, num=3):
    dataset_dicts = DatasetCatalog.get(name)
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = out.get_image()[:, :, ::-1]
        cv2.imshow(d["file_name"], img)
        cv2.waitKey(0)
