# ui/detection_view.py
import os
from PyQt5.QtWidgets import QWidget,QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox 
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtCore import QObject, pyqtSignal
import cv2

from models.instant_frame import Frame
from utils.logger import CustomLogger


class DetectionView(QWidget):
    en_demo = pyqtSignal(str, str)
    
    def __init__(self, device_name, device_attribute, camera_service, config, logger):
        super().__init__()
        self.device_name = device_name
        self.device_attribute = device_attribute
        self.lg = logger
        self.camera_service = camera_service
        self.pc_screen_size = QApplication.primaryScreen().size()
        self.config = config
        # scale image to 75% of original size since multiple 480p screen is too big for 1920x1080 screen
        self.img_scale = config.get('img_scale', 1)
        self.scaled_size = (int(640*self.img_scale), int(480*self.img_scale))

        self.setup_ui()

        for i in range(len(self.camera_service)):
            self.en_demo.connect(self.camera_service[i].on_demo_signal_received)
        
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info and controls layout
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"Device: {self.device_name}"))
        if self.device_attribute['device_type'] == 'dss':
            info_layout.addWidget(self.create_button("Force Ext", self.force_ext))
            info_layout.addWidget(self.create_button("Demo", self.demo))

        elif self.device_attribute['device_type'] == 'tr':
            info_layout.addWidget(self.create_button("Demo", self.demo))

        layout.addLayout(info_layout)

        # Camera layouts
        self.camera_labels = []
        camera_layouts = QHBoxLayout()
        
        for i, cam_idx in enumerate(self.device_attribute['cams']):
            cam_layout = QVBoxLayout()
            label = self.create_camera_label()
            self.camera_labels.append(label)
            cam_layout.addWidget(label)
            cam_layout.addWidget(QLabel(f"Camera {i+1} - {cam_idx}"))
            camera_layouts.addLayout(cam_layout)
        
        layout.addLayout(camera_layouts)
        
        # Connect frame received signal
        for i in range(len(self.camera_service)):
            self.camera_service[i].frame_received.connect(self.update_image, Qt.QueuedConnection)

    
    def create_button(self, text, callback):
        button = QPushButton(text)
        button.setMaximumWidth(150)
        button.clicked.connect(callback)
        return button
    
    def create_camera_label(self):
        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        
        self.set_placeholder_image(label)
        return label
    
    def set_placeholder_image(self, label):
        base_img = cv2.imread(os.path.join(os.getcwd(), 'ui', 'assets', 'stream_def.png'), cv2.IMREAD_COLOR)
        pixmap = self.cv_image_to_qpixmap(base_img).scaled(320, 240, Qt.KeepAspectRatio)

        if self.pc_screen_size.width() <= 1920 or True:
            pixmap = pixmap.scaled(self.scaled_size[0], self.scaled_size[1], Qt.KeepAspectRatio)

        label.setPixmap(pixmap)
    
    def force_ext(self):
        self.lg.info("Force Ext button clicked")
        # self.mqtt_service.force_ext(self.device_name)

    def demo(self):
        self.lg.info(f"demo enabled - {self.device_name}")
        self.en_demo.emit(self.device_attribute['device_type'],self.device_name)

    def update_image(self, frame_obj: Frame):
        if frame_obj.device_name != self.device_name:
            return
        
        pixmap = self.cv_image_to_qpixmap(frame_obj.frame)

        if self.pc_screen_size.width() <= 1920 or True:
            pixmap = pixmap.scaled(self.scaled_size[0], self.scaled_size[1], Qt.KeepAspectRatio)

        if frame_obj.cam_idx < len(self.camera_labels):
            self.camera_labels[frame_obj.cam_idx].setPixmap(pixmap)

        # print(f"frame received from {frame_obj.device_name} cam {frame_obj.cam_idx}")
    
    @staticmethod
    def cv_image_to_qpixmap(cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(q_img)

