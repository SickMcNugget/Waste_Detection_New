from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

import random
import cv2

def register_waste_dataset():
    register_coco_instances("trash_train", 
        {}, 
        "/home/joren/Documents/full_Trash_Dataset/train/_annotations.coco.json",
        "/home/joren/Documents/full_Trash_Dataset/train/")
    register_coco_instances("trash_valid",
        {},
        "/home/joren/Documents/full_Trash_Dataset/valid/_annotations.coco.json",
        "/home/joren/Documents/full_Trash_Dataset/valid/")
    register_coco_instances("trash_test",
        {},
        "/home/joren/Documents/full_Trash_Dataset/test/_annotations.coco.json",
        "/home/joren/Documents/full_Trash_Dataset/test/")


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