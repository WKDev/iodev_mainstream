import os
import torch
from ultralytics import YOLO
from utils.logger import CustomLogger


def load_model(config, lg: CustomLogger , dev_type: str = 'dss'):

    # check if GPU is available
    device = 'GPU'
    if not torch.cuda.is_available():
        lg.warning("CUDA is not available")
        device = 'CPU'

    pth = os.path.join(os.getcwd(), config['detection']['model_dir'], config['detection'][dev_type]['weights'])
    lg.debug(f"[{device}]Loading {dev_type} weights from {pth}")

    yolo = YOLO(pth, verbose=config['detection']['verbose'])

    return yolo
