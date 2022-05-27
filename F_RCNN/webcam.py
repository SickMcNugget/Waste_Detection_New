import cv2
import argparse
import os
from tqdm import tqdm
from detectron2.utils.logger import setup_logger

from waste_utils import WasteVisualizer
from waste_utils import waste_cfg

def main():
    parser = argparse.ArgumentParser(description="Use a webcam with a trained model")
    parser.add_argument("path", help="The path to the trained model")

    args = parser.parse_args()

    setup_logger()
    cfg = waste_cfg("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(args.path, "model_final.pth")

    vis = WasteVisualizer(cfg)
    cam = cv2.VideoCapture(0)
    for vis in tqdm(vis.run_on_video(cam)):
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.imshow("Webcam", vis)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
