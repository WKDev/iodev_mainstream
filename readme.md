# summary 
1. takes rtsp streams with opencv and feed the stream into yolo_service
2. yolo_service performs object detection, and shows the result
3. if there's any detection, publish current timestamp into "${device_name}/time" and write new log.

# environments
python 3.10
pyqt5
ultralytics yolov8

# features
- logging with following feature
    - file saving
    - sends logs into pyqt gui
    - colored
- configuration
    - if there's no configuration, use default configuration.
- mqtt 
    - if mqtt client fails to connect, add log then connect again in 1s.

- rtsp stream
    - if fails to open stream, show fallback image and try to connect again.
- yolo_service
    - save prev 10 InstantFrame into buffer.


# ui component
- main ui
    - [dss1 detection_view] | [dss2 detection_view]
    - [dss3 detection_view] | [dss4 detection_view]
    - [logs list_view]

- detection view
    - [device_name text ] |[rtsp_url text]
    - [cam1 image] | [cam1 image]
    - [Detection on/off switch] | [Force Ext button] | [Demo button]

# models (pseudo code)
class InstantFrame: # use this model when this app put streams into YoloService and pyqt receives the frame with bounding box.
    def __init__(self,name,time,frame):
        self.name = name
        self.time = time.time()
        self.frame = frame
        self.detections = []