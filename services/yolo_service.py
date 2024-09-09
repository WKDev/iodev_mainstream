import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from models.instant_frame import Frame
import os

class YoloService:
    def __init__(self, model_path, model_name):
        path = os.path.join(os.path.abspath(model_path), f"{model_name}")
        print(f"loading model from {path}")

        self.model = YOLO(path, task='detect', verbose=True)
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your needs
        self.frame_buffers = {}
    
    def process_frame(self, device_name, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.tolist()
        
        instant_frame = Frame(device_name, frame)
        instant_frame.detections = detections
        
        self._update_frame_buffer(device_name, instant_frame)
        
        return instant_frame
    
    def process_stream(self, device_name, stream):
        return self.executor.submit(self._process_stream_thread, device_name, stream)

    
    def _update_frame_buffer(self, device_name, instant_frame):
        if device_name not in self.frame_buffers:
            self.frame_buffers[device_name] = []
        
        self.frame_buffers[device_name].append(instant_frame)
        if len(self.frame_buffers[device_name]) > 10:
            self.frame_buffers[device_name].pop(0)