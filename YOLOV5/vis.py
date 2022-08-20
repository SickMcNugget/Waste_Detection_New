import math
from multiprocessing import Pool
import os
from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import random
import argparse
import sys

def main():

    # User input will determine the total number of images to save
    parser = argparse.ArgumentParser(description='Annotate and save a chosen number of images')
    parser.add_argument('--number', default=1, metavar='NUM', type=int,
                        help='The number of images to save')
    parser.add_argument('--data-path', default=".", metavar='DATA', help="The directory containing COCO images")
    parser.add_argument('--ann-path', metavar='ANN',
                        help='The annotation file for the data path')
    parser.add_argument('--show', help="Shows the plot instead of exporting the image", action='store_true')
    args = parser.parse_args()
    
    #The location of the images and annotations
    full_annotation_path = ""
    if args.ann_path is not None:
        full_annotation_path = args.ann_path
    else:
        full_annotation_path = os.path.join(args.data_path, "_annotations.coco.json")

    if os.path.exists(os.path.abspath(full_annotation_path)):
        coco_annotation = COCO(annotation_file=full_annotation_path)

        # Category IDs
        cat_ids = coco_annotation.getCatIds()
        assert len(cat_ids) != 0

        # All categories
        cats = coco_annotation.loadCats(cat_ids)
        cat_names = [cat["name"] for cat in cats]
        img_ids = coco_annotation.getImgIds()

        # Multiprocessing to save images faster
        total_pools = os.cpu_count()
        # Allows <total_pools> number of images to be saved simultaneously
        total_loops = math.ceil(args.number / total_pools)
        img_ids = random.sample(img_ids, args.number)

        # Processes and saves one image at a time
        for i in range(total_loops):
            pooldata = [
                (img_ids[img_id], coco_annotation, args.data_path, cat_names, args) 
                    for img_id in range(i*total_pools, min(total_pools * (i+1), args.number))
                ]
            with Pool(processes=total_pools) as pool:
                pool.starmap(annotate_and_save, pooldata)

def annotate_and_save(img_id, coco_annotation, data_path, cat_names, args):
    '''Uses a COCO image ID and annotation file to load in an image
    and then annotate the image with bounding boxes. The result is then
    saved to disk
    '''
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
    if args.show:
        plt.show()
    else:
        fig.savefig(f"imgs/{img_id}_annotated.jpg", dpi=600, bbox_inches="tight", pad_inches=0)

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