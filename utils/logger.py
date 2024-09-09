# utils/logger.py
import logging
import os
from queue import Queue
from PyQt5.QtCore import QObject, pyqtSignal
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QVBoxLayout, QWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import logging
import coloredlogs
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QListView, QVBoxLayout, QWidget
from collections import deque

MAX_LOG_ENTRIES = 500
class CustomLogger(QObject):
    log_signal = pyqtSignal(str, int, str, str)  # message, level, logger_name

    def __init__(self, name, config, log_file='app.log'):
        super().__init__()
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set log level from configuration
        log_level = config['log']['level']
        self.logger.setLevel(getattr(logging, log_level))
        
        self.log_buffer = deque(maxlen=MAX_LOG_ENTRIES)

        # File handler
        log_path = config['log']['path']
        os.makedirs(log_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler with colored logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        coloredlogs.install(level=log_level, logger=self.logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Custom handler to emit signal and store log
        class SignalHandler(logging.Handler):
            def __init__(self, signal, buffer, logger_name):
                super().__init__()
                self.signal = signal
                self.buffer = buffer
                self.logger_name = logger_name

            def emit(self, record):
                msg = self.format(record)
                self.buffer.append((msg, record.levelno, self.logger_name, record.asctime))
                self.signal.emit(msg, record.levelno, self.logger_name, record.asctime)

        signal_handler = SignalHandler(self.log_signal, self.log_buffer, self.name)
        signal_handler.setLevel(getattr(logging, log_level))
        self.logger.addHandler(signal_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def get_all_logs(self):
        return list(self.log_buffer)

class LogManager:
    def __init__(self, config):
        self.loggers = {}
        self.config = config

    def get_logger(self, name):
        if name not in self.loggers:
            self.loggers[name] = CustomLogger(name, self.config)
        return self.loggers[name]

    def get_all_logs(self):
        all_logs = []
        for logger in self.loggers.values():
            all_logs.extend(logger.get_all_logs())
        return sorted(all_logs, key=lambda x: x[0])  # Sort by message (which includes timestamp)