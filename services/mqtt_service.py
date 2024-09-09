import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt
import time
import os
from paho.mqtt.client import Client
class MqttService(QThread):
    def __init__(self, config, logger):
        super().__init__()
        self.lg = logger
        self.config = config
        self.client = mqtt.Client(client_id=config['client_id'])
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.lg.info("MQTT service started")
        self.lg.info(f"Connecting to MQTT broker at {config['host']}:{config['port']}")
        self.connect()

    
    def connect(self):
        while True:
            try:
                self.client.connect(self.config['host'], self.config['port'], self.config['keepalive'])
                self.client.loop_start()
                break
            except Exception as e:
                self.lg.error(f"Failed to connect to MQTT broker {self.config['host']}:{self.config['port']}: {e}")
                time.sleep(1)
    
    def publish(self, topic, message):
        self.client.publish(topic, message, qos=self.config['qos'])
    
    def _on_connect(self, client, userdata, flags, rc):
        self.lg.info(f"Connected to MQTT broker with result code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.lg.warning(f"Disconnected from MQTT broker with result code {rc}")
        self.connect()  # Attempt to reconnect

    def run_ext(self, device_name):
        self.lg.info(f"exterminating {device_name}")
        self.publish(f"ext/{device_name}", time.time()) 