import math
from multiprocessing import Pool
import os
from turtle import color
from unicodedata import category
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import random
import sys
import argparse

def main():

    # User input will determine the total number of images to save
    parser = argparse.ArgumentParser(description='Annotate and save a chosen number of images')
    parser.add_argument('number', metavar='N', type=int,
                        help='The number of images to save')
    parser.add_argument('--data-path', help="The directory containing COCO images")
    parser.add_argument('--ann-path', default=args.data_path + "_annotations.coco.json",
                        help='The annotation file for the data path')
    args = parser.parse_args()

    print(args.ann_path)

    #The location of the images and annotations
    data_path = "../datasets/TACO_TN_UAV_9-COCO_raw/train/"
    coco_annotation_file_path = data_path + "_annotations.coco.json"

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs
    cat_ids = coco_annotation.getCatIds()

    # All categories
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    img_ids = coco_annotation.getImgIds()

    # Multiprocessing to save images faster
    total_pools = os.cpu_count()
    # Allows <total_pools> number of images to be saved simultaneously
    total_loops = math.ceil(args.number / total_pools)
    img_ids = random.sample(img_ids, args.number)

    for i in range(total_loops):
        pooldata = [
            (img_ids[img_id], coco_annotation, data_path, cat_names) 
                for img_id in range(i*total_pools, min(total_pools * (i+1), args.number))
            ]
        with Pool(processes=total_pools) as pool:
            pool.starmap(save_random, pooldata)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def save_random(img_id, coco_annotation, data_path, cat_names):
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]

    # Get all the annotations for the specified image.
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)

    # Use URL to load image.
    im = Image.open(data_path + img_file_name)

    # Save image and its labeled version.
    fig,ax = plt.subplots()
    ax.axis("off")
    ax.imshow(np.asarray(im))

    # Plot segmentation and bounding box.
    coco_annotation.showAnns(anns, draw_bbox=True)
    put_text(cat_names, anns, np.asarray(im))
    fig.savefig(f"imgs/{img_id}_annotated.png", dpi=600, transparent=True)

def put_text(cat_names, anns, im):
    font = {
        'size': 12,
        'ha':'center', 
        'va': 'center'
    }

    for ann in anns:
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        pos = (bbox_x + (bbox_w / 2), bbox_y)
        label = cat_names[ann['category_id'] - 1]
        plt.text(pos[0], pos[1], label, font, color='yellow')

if __name__ == "__main__":
    main()