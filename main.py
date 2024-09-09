# main.py
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
# from services.yolo_service import YoloService
from services.mqtt_service import MqttService
from services.camera_service import CameraService
from utils.config_loader import load_config
from utils.logger import CustomLogger, LogManager
import sys
import torch
from ultralytics import YOLO
import os


def main():
    config = load_config()
    log_manager = LogManager(config)

    cs_logger = log_manager.get_logger('CameraService')
    mw_logger = log_manager.get_logger('MainWindow')
    ct_logger = log_manager.get_logger('CameraThread')
    mt_logger = log_manager.get_logger('MqttService')
    ml_logger = log_manager.get_logger('ModelLoader')



    app = QApplication(sys.argv)

    # mqtt_service = MqttService(config['mqtt'], mt_logger)
    # mqtt_service.start()
    
    mw = MainWindow(config, log_manager)
    for logger in log_manager.loggers.values():
        logger.log_signal.connect(mw.add_log_to_view)
    mw.show()
   
    app.exec_()
    
if __name__ == "__main__":
    main()