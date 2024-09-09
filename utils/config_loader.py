# utils/config_loader.py
import yaml
import os
from utils.logger import CustomLogger

def load_config(config_path='configuration.yaml'):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return default_config()
    else:
        print(f"Loading config from: {config_path}")

        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

def default_config():
    return {
        'detection': {
            'model_path': 'weights',
            'model_name': 'dss_model.pt',
            'conf_thres': 0.5,
            'iou_thres': 0.5,
            'device': 'cuda:0',
        },
        'log': {
            'level': 'INFO',
            'path': 'logs'
        },
        'mqtt': {
            'host': 'localhost',
            'port': 1883,
            'qos': 0,
            'keepalive': 60,
            'client_id': 'default_client'
        },
        'devices': {
            'dss1':
            {'device_type': "dss",
            'cams': [0]

            },
    }

        
    }