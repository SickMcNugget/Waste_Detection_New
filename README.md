## Repos being used
**detectron2** https://github.com/facebookresearch/detectron2  
**YOLOv5** https://github.com/ultralytics/yolov5  
**DETR** https://github.com/facebookresearch/detr  

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
## Training
### YOLOv5
```shell
cd YOLOv5  
python train.py --img 640 --batch 16 --data trash.yaml --weights yolov5m.pt  
```
## Contribute
![Alt text](../assets/example.png?raw=true)
