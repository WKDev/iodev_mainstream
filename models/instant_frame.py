from collections import defaultdict
import time

class Frame:
    def __init__(self, name, cam_idx, source, frame):
        self.device_name = name
        self.cam_idx = cam_idx
        self.source = source
        self.time = time.time()
        self.frame = frame
        self.detections = []
        self.track_history = defaultdict(lambda: [])
