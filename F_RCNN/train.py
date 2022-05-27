import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# detectron2
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
setup_logger()

# built-in libraries
from datetime import datetime
import os

# custom
from waste_utils import register_waste_dataset
from waste_utils import waste_cfg

#Make sure detectron knows where the datasets are
register_waste_dataset()
#Load some default config parameters
cfg = waste_cfg()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
