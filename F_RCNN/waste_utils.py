from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer

import random
import cv2

def show_dataset(name, num):
    dataset_dicts = DatasetCatalog.get(name)
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
        cv2.waitKey(0)