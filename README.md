- [Repos being used](#repos-being-used)
- [Installation](#installation)
  - [Begin with detectron2 (for linux)](#begin-with-detectron2-for-linux)
  - [YOLOv5](#yolov5)
  - [DETR](#detr)
- [Training](#training)
  - [YOLOv5](#yolov5-1)
- [Contribute](#contribute)
  - [Classes (case sensitive)](#classes-case-sensitive)
  - [Submission Format](#submission-format)
  - [Notes](#notes)
  - [Example Annotation](#example-annotation)

---

## Repos being used
**detectron2** https://github.com/facebookresearch/detectron2  
**YOLOv5** https://github.com/ultralytics/yolov5  
**DETR** https://github.com/facebookresearch/detr  

---

## Installation
**Please create a conda environment to make installation easier**
```shell
conda create --name waste python=3.9 -y  
conda activate waste  
```
**Depending on the version of CUDA that is required the pytorch installation will vary. Change cudatoolkit=11.3 to 10.2 if CUDA 11.3 is not supported on your GPU**
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch  
pip install opencv-python  
```
### Begin with detectron2 (for linux)
```shell
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'  
```
### YOLOv5
```shell
git clone https://github.com/ultralytics/yolov5  
cd yolov5  
pip install -r requirements.txt
pip install wandb
```
### DETR
```shell
git clone https://github.com/facebookresearch/detr.git  
conda install cython scipy  
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'  
```

---

## Training
### YOLOv5
```shell
cd YOLOV5  
python train.py --img 640 --batch 16 --data trash_15.yaml --weights yolov5m.pt --hyp hyp.scratch-low_trash.yaml  
OR  
python train.py --img <dimensions> --batch <batch_size recc. 64> --data <yaml for dataset> --weights yolov5m.pt --hyp <customised hyperparameters>
```

---

## Contribute
### Classes (case sensitive)
There are 15 classes. 'negative' represents non-trash.
- aluminium wrap
- cardboard
- cigarette
- foil
- general waste
- glass
- metal
- negative
- paper
- plastic bag
- plastic bottle
- plastic cup
- plastic straw
- plastic
- styrofoam

### Submission Format
Please submit annotations in the [COCO Format](https://cocodataset.org/#format-data)  
The annotation type is "1. Object Detection" on the same page.  

### Notes
If an object is made of thin cardboard (between paper and cardboard), opt for 'paper'.  
If you can't tell whether an object is a certain material, opt for 'general waste'.  
When annotating an image, be sure to make the bounding box as small as possible.

### Example Annotation
![Alt text](../assets/example.png?raw=true)
