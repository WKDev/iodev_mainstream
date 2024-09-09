from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListView, QTabWidget
from PyQt5.QtCore import QStringListModel

from services.camera_service import CameraService
from utils.logger import CustomLogger
from .detection_view import DetectionView
import logging
import coloredlogs
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QVBoxLayout, QWidget
import time

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

MAX_LOG_ENTRIES = 500


class MainWindow(QMainWindow):
    def __init__(self, config, log_manager):
        super().__init__()
        self.config = config

        self.log_manager = log_manager
        
        self.setWindowTitle("SID")
        self.screen_size = QApplication.primaryScreen().size()
        self.setGeometry(int(self.screen_size.width()/2), int(self.screen_size.height()/2), 1600, 900)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        # self.devices_layout = QHBoxLayout()

        # self.dss_layout = QGridLayout()  # QGridLayout 사용
        # self.tr_layout = QGridLayout()

        # self.devices_layout.addLayout(self.dss_layout)
        # self.devices_layout.addLayout(self.tr_layout)
        # self.main_layout.addLayout(self.devices_layout)

        self.camera_services = {}
    
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create separate tabs for DSS and TR devices
        self.dss_tab = QWidget()
        self.tr_tab = QWidget()

        self.dss_layout = QGridLayout(self.dss_tab)
        self.tr_layout = QGridLayout(self.tr_tab)
        self.tab_widget.addTab(self.dss_tab, "조류퇴치기")
        self.tab_widget.addTab(self.tr_tab, "디지털 트랩")


        self.init_camera_service()

        self.setup_detection_views()
        self.setup_log_view()
        self.log_manager = log_manager


    def init_camera_service(self):
        for device_name, device_attribute in self.config['devices'].items():
            self.camera_services[device_name] = []
            for idx in range(len(device_attribute['cams'])):
                self.camera_services[device_name].append(CameraService(device_attribute['device_type'],
                                                                device_name,
                                                idx, 
                                                self.config, 
                                                self.log_manager.get_logger(f"CameraService_{device_attribute['device_type']}_{device_name}_{idx}")
                                                ))
                self.camera_services[device_name][idx].start()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_size_label()


    def update_size_label(self):
        size = self.size()
        print(f"창 크기 변경: 너비 = {size.width()}, 높이 = {size.height()}")

    
    def setup_detection_views(self):
        devices = list(self.config['devices'].items())
        num_devices = len(devices)

        # 4개 이하면 2x2, 5개 이상이면 4x2
        max_columns = 2

        dss_grid = [0,0]
        tr_grid = [0,0]

        for device_name, device_attribute in devices:
            # print(f"Setting up detection view for device: {device_name} | {device_attribute}")
            if device_attribute['device_type'] == 'dss':
                detection_view = DetectionView(device_name, device_attribute, self.camera_services[device_name], self.config['display'],self.log_manager.get_logger("DetectionView-dss"))
                # 해당 그리드 위치에 위젯 추가
                self.dss_layout.addWidget(detection_view, dss_grid[0], dss_grid[1])

                dss_grid[1] += 1
                if dss_grid[1] >= max_columns:  # 최대 열 개수에 도달하면 다음 행으로
                    dss_grid[1] = 0
                    dss_grid[0] += 1

            elif device_attribute['device_type'] == 'tr':
                detection_view = DetectionView(device_name, device_attribute, self.camera_services[device_name], self.config['display'],self.log_manager.get_logger("DetectionView-tr"))
                # 해당 그리드 위치에 위젯 추가
                self.tr_layout.addWidget(detection_view, tr_grid[0], tr_grid[1])

                tr_grid[1] += 1
                if tr_grid[1] >= max_columns:  # 최대 열 개수에 도달하면 다음 행으로
                    tr_grid[1] = 0
                    tr_grid[0] += 1


    
    def setup_log_view(self):
        self.log_view = QListView()
        # use dark gray background color
        self.log_view.setStyleSheet("background-color: #888888; color: #ffffff")


        self.main_layout.addWidget(self.log_view)
        self.log_model = QStandardItemModel()
        self.log_view.setModel(self.log_model)

    def add_log_to_view(self, message, level, logger_name, asctime):
        item = QStandardItem(f"{asctime} {logger_name}: {message}")
        if level == logging.DEBUG:
            item.setForeground(Qt.blue)
        elif level == logging.INFO:
            item.setForeground(Qt.white)
        elif level == logging.WARNING:
            item.setForeground(Qt.yellow)
        elif level == logging.ERROR:
            item.setForeground(Qt.red)
        elif level == logging.CRITICAL:
            item.setForeground(Qt.darkRed)
        
        # self.log_model.insertRow(0, item)        
        self.log_model.appendRow(item)

        while self.log_model.rowCount() > MAX_LOG_ENTRIES:
            self.log_model.removeRow(self.log_model.rowCount() - 1)

        self.log_view.scrollToBottom()

    def showEvent(self, event):
        super().showEvent(event)

        for msg, level, logger_name, asctime in self.log_manager.get_all_logs():
            self.add_log_to_view(msg, level, logger_name, asctime)
