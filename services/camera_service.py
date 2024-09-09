# read mutiple camera streams concorrently and save them into its member variable

from collections import defaultdict, deque
import glob
from PIL import Image
from typing import Dict, List
import uuid
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt
import time
import os
from models.instant_frame import Frame
from utils.demo import optimized_random_image_placement, overlay_birds_on_frame, random_image_placement
from utils.logger import CustomLogger
from threading import Thread
from ultralytics import YOLO
import torch
from sklearn.neighbors import KDTree


from utils.model import load_model

class DetectionState:
    def __init__(self):
        self.frame_buffer = deque(maxlen=5)
        self.is_detecting = False
        self.ext_run_executed = False

        self.detection_start_time = 0
        self.last_detection_time = 0
        self.static_frame_count = 0
        self.detection_frame_count = 0
        self.active_objects = {}

        self.prev_tree = None
        self.prev_objects = None

class CameraService(QThread):
    frame_received = pyqtSignal(object)

    def __init__(self,  dev_type:str, dev_name: str, cam_idx, config :dict, cs_logger, **kwargs):
        super().__init__()
        self.dev_type = dev_type
        self.dev_name = dev_name
        self.cam_idx = cam_idx

        self.lg = cs_logger
        self.lg.debug(f"Camera service started : {self.dev_type} {self.dev_name} {self.cam_idx}")

        self.config = config
        self.threads = []
    
        self.detection_config = config['detection'][self.dev_type]
        self.demo_config = config['demo'][self.dev_type]
        self.max_detection_duration = self.detection_config['max_detection_duration']  # 최소 탐지 지속 시간 (초)
        self.similarity_tol = self.detection_config['similarity_tol']  # 오브젝트 유사성 허용치

        if kwargs.get('model') is None:
            self.lg.debug("loading model inside CameraService")
            self.model = load_model(self.config, self.lg, dev_type=self.dev_type)

        self.detection_state = DetectionState()

        # Load birds images for demo mode
        if self.dev_type == 'dss':
            base_paths = glob.glob(os.path.join(os.getcwd(), 'demo','base_birds_alpha', '*.png'))
        elif self.dev_type == 'tr':
            base_paths = glob.glob(os.path.join(os.getcwd(), 'demo','base_bugs', '*.png'))


        self.demo_base_images = [Image.open(path).convert('RGBA') for path in base_paths]

        self.demo_base_image= None


        # demo mode
        self.demo_mode = False
        self.demo_devices = []
        self.demo_frame_count = 0
        self.demo_max_frames = 180
        self.demo_mask = None
        self.demo_offset = 0


    def run(self):
        self.capture_camera_stream(self.dev_name)

    def capture_camera_stream(self, dev_name):
        cam_source = self.config['devices'][dev_name]['cams'][self.cam_idx]
        self.lg.debug(f"Capturing camera stream from {cam_source}")

        img_def = cv2.imread(os.path.join(os.getcwd(), 'ui', 'assets', 'stream_def.png'))
        img_ready = cv2.imread(os.path.join(os.getcwd(), 'ui', 'assets', 'stream_ready.png'))

        self.frame_received.emit(Frame(dev_name, self.cam_idx, cam_source, img_def))

        cap = None

        while True:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(cam_source)
                if not cap.isOpened():
                    self.lg.error(f"Failed to open camera {cam_source}")
                    self.frame_received.emit(Frame(dev_name, self.cam_idx, cam_source, img_ready))
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                self.lg.error(f"Error reading frame from camera {cam_source}")
                cap.release()
                cap = None
                continue

            frame_obj = Frame(dev_name, self.cam_idx, cam_source, frame)
            self.handle_frame(frame_obj)

            if self.isInterruptionRequested():
                break

        if cap is not None:
            cap.release()
        self.lg.debug(f"Camera stream capture ended for {cam_source}")
      
    def handle_frame(self, frame_obj: Frame):
        if self.demo_mode:
            if self.demo_frame_count < self.demo_max_frames:
                if frame_obj.device_name in self.demo_devices:
                    frame_obj.frame = self.apply_demo_effect(frame_obj.frame)
                self.demo_frame_count += 1
                self.demo_offset += self.demo_config['move_displacement']  
            else:
                self.demo_mode = False
                self.demo_devices=[]

                self.lg.info("Demo mode completed")

        if frame_obj.frame.shape[2] == 4:  # RGBA
            frame_obj.frame = cv2.cvtColor(frame_obj.frame, cv2.COLOR_RGBA2RGB)

        results = self.model.predict(frame_obj.frame, conf=self.detection_config['confidence_threshold'], iou=self.detection_config['iou_threshold'], verbose=False)
        
        if self.dev_type == 'dss':
            frame_detected = self.process_detection(frame_obj, results)
            self.frame_received.emit(frame_detected)


        elif self.dev_type == 'tr':
            frame_obj.frame = results[0].plot(labels=False, line_width=self.detection_config['bbox_line_width'])
            self.frame_received.emit(frame_obj)

        

    def on_demo_signal_received(self, device_type, target):
        self.lg.info(f"Demo signal received: {device_type=} {target=}")
        
        if device_type == self.dev_type:
            self.demo_devices.append(target)
            self.demo_mode = True
            self.demo_frame_count = 0
            self.demo_offset = 0

            background = np.zeros((480, 640, 4), dtype=np.uint8)  # 640x480 frame with RGBA (4 channels for transparency)
            background_pil = Image.fromarray(background, 'RGBA')

            self.demo_base_image = optimized_random_image_placement(
                background_size=background_pil.size,  
                images=self.demo_base_images,
                **self.demo_config                    
            )

            self.demo_base_image = np.array(self.demo_base_image)

        else:
            self.demo_devices = []
            self.demo_mode = False
            self.demo_base_image = None
        # self.mqtt_service.demo(self.device_name, enabled)

    def apply_demo_effect(self, frame):
        if self.demo_base_image is None:
            return frame

        height, width = frame.shape[:2]
        birds_height, birds_width = self.demo_base_image.shape[:2]

        # Calculate the offset for x-axis movement
        x_offset = self.demo_offset % width

        large_birds_image = np.zeros((height, width * 2, 4), dtype=np.uint8)
        large_birds_image[:birds_height, :birds_width] = self.demo_base_image
        large_birds_image[:birds_height, birds_width:birds_width * 2] = self.demo_base_image
    # Crop the birds image with the current offset
        moving_birds = large_birds_image[:, x_offset:x_offset+width]

        # Overlay the moving birds on the frame
        result = overlay_birds_on_frame(frame, moving_birds)

        return result

    def process_detection(self, frame_obj: Frame, results) -> Frame:
        current_time = time.time()
        frame = frame_obj.frame
        height, width = frame.shape[:2]
        device_name = frame_obj.device_name

        # 1. Preprocessing
        moving_objects, valid_objects, invalid_objects = self.preprocess_objects(frame_obj,results, width, height)

        # 2. Start condition check
        if not self.detection_state.is_detecting:
            if self.check_start_condition(frame_obj, current_time, valid_objects):
                self.start_detection(current_time, frame_obj)

        # 3. Run detection
        if self.detection_state.is_detecting:
            self.detection_state.detection_frame_count += 1
            
            # Execute run_ext only once when detection starts
            if not self.detection_state.ext_run_executed:
                self.lg.info("Executing run_ext for the first time in this detection period")
                self.run_ext(frame_obj, valid_objects)
                self.detection_state.ext_run_executed = True
            
            # Log only every 100 frames or so
            if self.detection_state.detection_frame_count % 100 == 0:
                self.lg.info(f"Detection ongoing. Frame count: {self.detection_state.detection_frame_count}")

        # 4. End condition check
        if self.detection_state.is_detecting:
            if self.check_end_condition(frame_obj,current_time, valid_objects):
                self.end_detection(frame_obj)

        # Draw bounding boxes and other information on the frame
        if self.detection_config['show_bbox']:
            if self.detection_config['show_valid_detections']:
                frame = self.draw_results(frame_obj, valid_objects, color=(0,255,0), moving_objects=moving_objects)
            
            if self.detection_config['show_invalid_detections']:
                frame = self.draw_results(frame_obj, invalid_objects, color=(0,0,255))


        # 주기적으로 오래된 프레임 제거
        self.detection_state.frame_buffer.append(valid_objects)

        return Frame(frame_obj.device_name, frame_obj.cam_idx, frame_obj.source, frame)

    def preprocess_objects(self, frame_obj: Frame, results, width: int, height: int):
        moving_objects=[]
        valid_objects = []
        invalid_objects = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            obj_width, obj_height = x2 - x1, y2 - y1
            
            # Check if object size is less than 20% of the frame
            is_valid_size = self.detection_config['min_object_size_percent'] < (obj_width * obj_height) / (width * height) <= self.detection_config['max_object_size_percent']
            # Check if object has valid aspect ratio
            aspect_ratio = obj_width / obj_height
            is_valid_aspect_ratio = self.detection_config['min_aspect_ratio'] < aspect_ratio < self.detection_config['max_aspect_ratio']

            object_info = {
                'id': int(box.id) if box.id is not None else -1,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class': int(box.cls),
                'conf': float(box.conf)
            }

            if is_valid_size and is_valid_aspect_ratio:
                valid_objects.append(object_info)
            else:
                invalid_objects.append(object_info)

        # Remove objects with similar size and position to previous frame
        if self.detection_state.frame_buffer and self.detection_config['remove_similar_objects']:
            prev_objects = self.detection_state.frame_buffer[-1]
            moving_objects = self.remove_similar_objects( valid_objects, prev_objects)

        self.detection_state.frame_buffer.append(valid_objects)


        return moving_objects, valid_objects, invalid_objects

    def remove_similar_objects(self, current_objects: List[Dict], prev_objects: List[Dict]) -> List[Dict]:
        if not current_objects or not prev_objects:
            return current_objects

        filtered_objects = []

        for curr_obj in current_objects:
            if not any(self.is_object_similar_simple(curr_obj, prev_obj, self.similarity_tol) for prev_obj in prev_objects):
                filtered_objects.append(curr_obj)

        return filtered_objects
    
    def calculate_distance(self, obj1: Dict, obj2: Dict) -> float:
        center1 = ((obj1['bbox'][0] + obj1['bbox'][2]) / 2, (obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        center2 = ((obj2['bbox'][0] + obj2['bbox'][2]) / 2, (obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    def is_object_similar(self, obj1: Dict, obj2: Dict, distance: float) -> bool:
        size1 = (obj1['bbox'][2] - obj1['bbox'][0]) * (obj1['bbox'][3] - obj1['bbox'][1])
        size2 = (obj2['bbox'][2] - obj2['bbox'][0]) * (obj2['bbox'][3] - obj2['bbox'][1])
        
        size_similarity = abs(size1 - size2) < self.similarity_tol * max(size1, size2)
        distance_similarity = distance < self.similarity_tol * max(size1, size2)**0.5
        return size_similarity and distance_similarity
    

    def is_object_similar_simple(self, obj1: Dict, obj2: Dict, similarity_tol: float) -> bool:
        center1 = ((obj1['bbox'][0] + obj1['bbox'][2]) / 2, (obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        center2 = ((obj2['bbox'][0] + obj2['bbox'][2]) / 2, (obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        return distance < similarity_tol
    
    def check_start_condition(self, frame_obj, current_time: float, valid_objects: List[Dict]) -> bool:
        if current_time - self.detection_state.last_detection_time < self.detection_config['min_time_between_detections']:
            # self.lg.info(f"Detection start condition not met: Minimum time between detections not reached")

            return False

        # active_objects = self.get_active_objects(valid_objects)
        active_objects = valid_objects
        if len(active_objects) >= self.detection_config['min_object_cnt']:
            self.lg.info(f"Detection start condition met: {len(active_objects)} active objects detected")
            return True
        return False
    
    def start_detection(self, current_time: int, frame_obj: Frame):
        self.detection_state.is_detecting = True
        self.detection_state.detection_start_time = current_time
        self.detection_state.detection_frame_count = 0
        self.detection_state.ext_run_executed = False
        self.lg.info(f"[{frame_obj.device_name}][{frame_obj.cam_idx}]Detection started at {current_time}")

    def check_end_condition(self, frame_obj, current_time: float, valid_objects: List[Dict]) -> bool:
        # 최소 탐지 지속 시간 확인
        if current_time - self.detection_state.detection_start_time < self.max_detection_duration:
            return False

        # active_objects = self.get_active_objects(valid_objects)
        active_objects = valid_objects
        
        if len(active_objects) < self.detection_config['min_object_cnt']:
            self.detection_state.static_frame_count += 1
        else:
            self.detection_state.static_frame_count = 0

        if self.detection_state.static_frame_count >= self.detection_config['static_frame_threshold']:
            self.lg.info(f"Detection ended: Less than {self.detection_config['min_object_cnt']} active objects for {self.detection_config['static_frame_threshold']:.1f} frames")
            return True

        if current_time - self.detection_state.detection_start_time > self.detection_config['max_detection_duration']:
            self.lg.info(f"Detection ended: Maximum duration of {self.detection_config['max_detection_duration']:.1f} seconds reached")
            return True

        return False
    
    def run_detection(self, frame_obj: Frame, valid_objects: List[Dict]):
        self.detection_state.detection_frame_count += 1
        self.lg.info(f"Running detection on frame {self.detection_state.detection_frame_count}")
        self.run_ext(frame_obj, valid_objects)

    def run_ext(self, frame_obj: Frame, valid_objects: List[Dict]):
        # Placeholder for external detection logic
        self.lg.info("Running external detection logic")
        # Implement your specific detection logic here

    def end_detection(self,frame_obj: Frame):
        self.detection_state.is_detecting = False
        self.detection_state.last_detection_time = time.time()
        self.detection_state.static_frame_count = 0
        self.detection_state.ext_run_executed = False  # Reset the flag when ending detection
        self.lg.info(f"Detection ended in {self.detection_state.last_detection_time- self.detection_state.detection_start_time :.1f} seconds.")

    def draw_results(self, frame_obj : Frame, valid_objects: List[Dict], **kwargs) -> np.ndarray:

        color = kwargs.get('color', (0, 0, 0))
        moving_objects = kwargs.get('moving_objects', None)

        if moving_objects is not None:
            # draw dot on the center of moving objects in red
            for obj in moving_objects:
                x1, y1, x2, y2 = obj['bbox']
                x, y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame_obj.frame, (x, y), 5, (0, 0, 255), -1)


        for obj in valid_objects:
            x1, y1, x2, y2 = obj['bbox']
            # color = (0, 255, 0) if obj['id'] in self.active_objects else (0, 0, 255)
            cv2.rectangle(frame_obj.frame, (x1, y1), (x2, y2), color, self.detection_config['bbox_line_width'])
            # label = f"ID:{obj['id']} Class:{obj['class']} Conf:{obj['conf']:.2f}"
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if self.detection_state.is_detecting:
            cv2.rectangle(frame_obj.frame, (10, 10), (130, 30), (0, 0, 0), -1)
            cv2.putText(frame_obj.frame, "DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame_obj.frame
    



class CameraThread(QThread):
    raw_frame = pyqtSignal(object)
    def __init__(self, dev_name, cam_idx ,addr, logger):

        self.img_def = cv2.imread(os.path.join(os.getcwd(),'ui','assets','stream_def.png'))
        self.img_ready = cv2.imread(os.path.join(os.getcwd(),'ui','assets','stream_ready.png'))
        super().__init__()
        self.dev_name = dev_name
        self.addr = addr
        self.lg = logger
        self.cam_idx = cam_idx

        if dev_name.startswith('dss'):
            # load the camera matrix and dist_coeffs from dss_intrinsics_480p.npz file
            intrinsics_file = os.path.join(os.getcwd(), 'calibration', 'dss_intrinsics_480p.npz')
            intrinsics = np.load(intrinsics_file)
            self.camera_matrix = intrinsics['camera_matrix']
            self.dist_coeffs = intrinsics['dist_coeffs']
        else:
            self.camera_matrix = None
            self.dist_coeffs = None    

    def run(self):
        self.lg.info(f"dss device found: {self.dev_name}")
        self.raw_frame.emit(Frame(self.dev_name, self.cam_idx, self.addr, self.img_def))
        cap = None

        while True:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.addr)
                if not cap.isOpened():
                    self.lg.error(f"Failed to open camera {self.addr}")
                    self.raw_frame.emit(Frame(self.dev_name,self.cam_idx, self.addr, self.img_ready))

                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            self.raw_frame.emit(Frame(self.dev_name, self.cam_idx, self.addr, frame))

            if not ret:
                self.lg.error(f"Error reading frame from camera {self.addr}, {self.dev_name}")
                cap.release()
                cap = None
                continue
            
        # 프레임 처리 로직 실행
        cap.release()
        self.lg.error(f"Camera {self.addr} disconnected. Reconnecting...")
        time.sleep(1)

    def draw_results(self, frame, results):
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls)
                class_name = results[0].names[cls]
                
                conf = float(box.conf)
                
                track_id = int(box.id) if box.id is not None else None

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                if track_id is not None:
                    label += f" ID:{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame
    




    # def handle_frame_track(self, frame_obj : Frame):
    
    #     # results = self.model.track(frame_obj.frame, persist=True, show=False)
    #     results = self.model.track(frame_obj.frame, conf=0.5, iou=0.45)
    #     # annotated_frame = results[0].plot(line_width=1, font_size=0.5, labels=False)
    #     # annotated_frame = self.draw_results(frame_obj.frame, results)
    #     # Visualize the results on the frame
    #     annotated_frame = results[0].plot()

    #     # Plot the tracks
    #     if results[0].boxes.id != None:
    #         boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    #         ids = results[0].boxes.id.cpu().numpy().astype(int)
    #         # confidences = results[0].boxes.conf.cpu().numpy().astype(int)
            
    #         for box, track_id in zip(boxes, ids):
    #             x, y, w, h = box
    #             track = self.track_history[frame_obj.device_name][frame_obj.source][track_id]
    #             track.append((float(x+w/2), float(y+h/2)))  # x, y center point
    #             if len(track) > 30:  # retain 90 tracks for 90 frames
    #                 track.pop(0)

    #             # Draw the tracking lines
    #             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    #             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    #     frame_obj.frame = annotated_frame
    #     self.frame_received.emit(frame_obj)

