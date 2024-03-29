# Waste_Detection_New  

  - [1. Repos being used](#1-repos-being-used)
  - [2. Installation](#2-installation)
    - [2.1. Windows Only (Extra Step)](#21-windows-only-extra-step)
    - [2.2. All platforms](#22-all-platforms)
      - [2.2.1. detectron2](#221-detectron2)
        - [2.2.1.1. Windows extra install](#2211-windows-extra-install)
      - [2.2.2. YOLOv5](#222-yolov5)
      - [2.2.3. DETR](#223-detr)
  - [3. Training](#3-training)
    - [3.1. YOLOv5](#31-yolov5)
    - [3.2. Faster-RCNN and DETR](#32-faster-rcnn-and-detr)
  - [4. Contribute](#4-contribute)
    - [4.0. Current Datasets](#40-current-datasets)
    - [4.1. Classes (case sensitive)](#41-classes-case-sensitive)
    - [4.2. Submission Format](#42-submission-format)
    - [4.3. Notes](#43-notes)
    - [4.4. Example Annotation](#44-example-annotation)
  - [5. Milestones](#5-milestones)
    - [5.1. Data Statistics](#51-data-statistics)
    - [5.2. Metrics and Graphs](#52-metrics-and-graphs)

---

## 1. Repos being used
**detectron2** https://github.com/facebookresearch/detectron2  
**YOLOv5** https://github.com/ultralytics/yolov5  
**DETR** https://github.com/facebookresearch/detr  

---

## 2. Installation
### 2.1. Windows Only (Extra Step)
**Install Microsoft C++ Build Tools** https://visualstudio.microsoft.com/visual-cpp-build-tools/  
**Install Instructions** https://stackoverflow.com/a/64262038  
**Don't forget to restart your system**  
  
### 2.2. All platforms
**Install Miniconda from** https://docs.conda.io/en/latest/miniconda.html  
**Please create a conda environment to make installation easier**
```shell
conda create --name waste python=3.9 -y  
conda activate waste  
```
**Always make sure pip is up to date**
```shell
pip install --upgrade pip  
```
**Depending on the version of CUDA that is required the pytorch installation will vary. Change cudatoolkit=11.3 to 10.2 if CUDA 11.3 is not supported on your GPU**
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y  
pip install opencv-python  
```
#### 2.2.1. detectron2
```shell
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'  
```
##### 2.2.1.1. Windows extra install
```shell
conda install pywin32 -y
```
#### 2.2.2. YOLOv5
```shell
cd YOLOV5  
pip install -r requirements.txt  
pip install wandb  
cd ..  
```
#### 2.2.3. DETR
```shell
conda install cython scipy -y  
```

---

## 3. Training
### 3.1. YOLOv5
```shell
cd YOLOV5  
python train.py --img 640 --batch 16 --data trash_9.yaml --weights yolov5m.pt --hyp hyp.scratch-low_trash.yaml  
```
OR  
```shell
python train.py --img <dimensions> --batch <batch_size recc. 64> --data <yaml for dataset> --weights yolov5m.pt --hyp <customised hyperparameters>
```  
Finally, test the model  
```shell
python val.py --img 640 --batch 32 --data trash_9.yaml --weights ./runs/train/<exp#>/weights/best.pt --task test --device 0
```
### 3.2. Faster-RCNN and DETR
```shell
cd FRCNN_DETR
# Linux
./frcnn_1_gpu.sh (assuming the current dataset is being used)
# All
python train.py --type ('frcnn' or 'detr') --num-gpus=1 SOLVER.IMS_PER_BATCH <batch_size> SOLVER.BASE_LR <learning rate>
```
OR FOR MULTI-GPU
```shell
# Linux
./frcnn_4_gpu.sh
./detr_4_gpu.sh
# ALL
python train.py --type ('frcnn' or 'detr') --num-gpus=4 SOLVER.IMS_PER_BATCH <total_batch_size> SOLVER.BASE_LR <learning rate>
```
---

## 4. Contribute
### 4.0. Current Datasets
**TrashNet** (annotated newly with bounding boxes): https://github.com/garythung/trashnet  
**TACO** (annotated newly once again): http://tacodataset.org/  
**UAVVaste** (mapped to correct classes): https://uavvaste.github.io/  
  
### 4.1. Classes (case sensitive)
There are 9 classes.
- aluminium wrap
- cardboard
- cigarette
- general waste
- glass
- metal
- paper
- plastic
- styrofoam

### 4.2. Submission Format
Please submit annotations in the [COCO Format](https://cocodataset.org/#format-data)  
The annotation type is "1. Object Detection" on the same page.  

### 4.3. Notes
If an object is made of thin cardboard (between paper and cardboard), opt for 'paper'.  
If you can't tell whether an object is a certain material, opt for 'general waste'.  
When annotating an image, be sure to make the bounding box as small as possible.

### 4.4. Example Annotation
![Alt text](../assets/example.png?raw=true)

---

## 5. Milestones
### 5.1. Data Statistics
The current best performance was achieved by using a combination of TACO, TrashNet and UAVVaste.  
The dataset split used was:  
- Total images: 6460
- Train: ~4500 images (70%)
- Validation: 654 images (10%)
- Test: ~1300 images (20%)  

### 5.2. Metrics and Graphs
- mAP: **51.3%**
- Precision: **62.4%**
- Recall: **49.1%**  

![Alt text](../assets/results.png?raw=true)
