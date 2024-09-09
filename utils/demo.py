import glob
import os
import cv2
import numpy as np
import random
import cv2
import numpy as np
import random
import os

def overlay_birds_on_frame(frame, birds_image):
    """
    Overlay the synthesized birds image on the given frame.
    
    :param frame: Original frame (BGR format)
    :param birds_image: Synthesized birds image with alpha channel
    :return: Combined image
    """
    # Convert frame to BGRA
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    # Ensure both images have the same size
    if frame.shape[:2] != birds_image.shape[:2]:
        birds_image = cv2.resize(birds_image, (frame.shape[1], frame.shape[0]))
    
    # Combine images
    alpha_birds = birds_image[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_birds
    
    for c in range(3):  # RGB channels
        frame[:, :, c] = alpha_birds * birds_image[:, :, c] + alpha_frame * frame[:, :, c]
    
    return frame

def load_birds_data(object_path):
    """
    Load bird images from the specified directory.
    
    :param object_path: Path to the directory containing bird images
    :return: List of tuples, each containing (bird_path, bird_image)
    """
    birds_paths = glob.glob(os.path.join(object_path, '*.png'))
    birds_data = []

    for bird_path in birds_paths:
        bird_img = cv2.imread(bird_path, cv2.IMREAD_UNCHANGED)
        if bird_img is None:
            print(f"Failed to load image: {bird_path}")
            continue
        birds_data.append((bird_path, bird_img))

    return birds_data


import numpy as np
import random
from PIL import Image, ImageDraw
import math

def optimized_random_image_placement(background_size, images, **kwargs):
    """
    이미지를 효율적으로 무작위 배치하는 최적화된 함수.

    Parameters:
        background_size (tuple): 출력 배경 해상도 (width, height).
        images (list): PIL Image 객체가 포함된 리스트.
        n_images (int): 배치할 이미지 개수.
        min_size (int): 이미지의 최소 크기.
        max_size (int): 이미지의 최대 크기.
        grid_size (int): 그리드 셀의 크기.
        max_attempts (int): 이미지 배치 최대 시도 횟수.

    Returns:
        PIL.Image: 배치된 이미지를 포함하는 최종 이미지.
    """
    n_images = kwargs.get('n_images', 10)
    min_size = kwargs.get('min_size', 50)
    max_size = kwargs.get('max_size', 80)
    grid_size = kwargs.get('grid_size', 20)
    max_attempts = kwargs.get('max_attempts', 100)

    background = Image.new('RGBA', background_size, (255, 255, 255, 0))
    
    # 그리드 시스템 초기화
    grid_width = background_size[0] // grid_size
    grid_height = background_size[1] // grid_size
    grid = [[False for _ in range(grid_height)] for _ in range(grid_width)]

    def is_area_free(x, y, width, height):
        grid_x = x // grid_size
        grid_y = y // grid_size
        grid_w = (width + grid_size - 1) // grid_size
        grid_h = (height + grid_size - 1) // grid_size

        if grid_x + grid_w > grid_width or grid_y + grid_h > grid_height:
            return False

        return all(not grid[gx][gy] 
                   for gx in range(grid_x, grid_x + grid_w)
                   for gy in range(grid_y, grid_y + grid_h))

    def mark_area_used(x, y, width, height):
        grid_x = x // grid_size
        grid_y = y // grid_size
        grid_w = (width + grid_size - 1) // grid_size
        grid_h = (height + grid_size - 1) // grid_size

        for gx in range(grid_x, grid_x + grid_w):
            for gy in range(grid_y, grid_y + grid_h):
                grid[gx][gy] = True

    placed_images = 0
    for _ in range(n_images):
        img = random.choice(images)
        original_width, original_height = img.size
        scale_factor = random.uniform(min_size / min(original_width, original_height),
                                      max_size / max(original_width, original_height))
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        img = img.resize((new_width, new_height))

        angle = random.randint(0, 360)
        img = img.rotate(angle, expand=True)
        w, h = img.size

        for _ in range(max_attempts):
            x = random.randint(0, background_size[0] - w)
            y = random.randint(0, background_size[1] - h)
            
            if is_area_free(x, y, w, h):
                mark_area_used(x, y, w, h)
                background.paste(img, (x, y), img)
                placed_images += 1
                break
        
        if placed_images == n_images:
            break

    return background

def random_image_placement(background_size, images, **kwargs):
    """
    이미지를 무작위 크기, 위치, 회전으로 배치하여 겹치지 않도록 하는 함수.

    Parameters:
        background_size (tuple): 출력 배경 해상도 (width, height).
        images (list): PIL Image 객체가 포함된 리스트.
        n_images (int): 배치할 이미지 개수.
        min_size (int): 이미지의 최소 크기.
        max_size (int): 이미지의 최대 크기.

    Returns:
        PIL.Image: 배치된 이미지를 포함하는 최종 이미지.
    """
    n_images = kwargs.get('n_images', 10)  # 배치할 이미지 개수
    min_size = kwargs.get('min_size', 50)  # 이미지의 최소 크기
    max_size = kwargs.get('max_size', 80)  # 이미지의 최대 크기
    
    # 배경 이미지를 생성
    background = Image.new('RGBA', background_size, (255, 255, 255, 0))  # 투명 배경
    
    placed_boxes = []  # 이미 배치된 이미지의 영역을 저장하는 리스트
    
    for _ in range(n_images):
        # 이미지 무작위로 선택
        img = random.choice(images)
        
        # 원본 이미지의 가로세로 비율 유지하면서 크기 변환
        original_width, original_height = img.size
        scale_factor = random.uniform(min_size / min(original_width, original_height), 
                                      max_size / max(original_width, original_height))
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        img = img.resize((new_width, new_height))
        
        # 이미지 무작위로 회전
        angle = random.randint(0, 360)
        img = img.rotate(angle, expand=True)
        
        # 이미지의 크기 확인
        w, h = img.size
        
        # 겹치지 않도록 랜덤 위치 찾기
        while True:
            x = random.randint(0, background_size[0] - w)
            y = random.randint(0, background_size[1] - h)
            new_box = (x, y, x + w, y + h)
            
            # 다른 이미지와 겹치는지 확인
            overlap = any(intersect(new_box, placed_box) for placed_box in placed_boxes)
            
            if not overlap:
                placed_boxes.append(new_box)
                background.paste(img, (x, y), img)
                break
    
    return background

def intersect(box1, box2):
    """
    두 박스가 겹치는지 여부를 확인하는 함수.

    Parameters:
        box1 (tuple): 첫 번째 박스의 좌표 (x1, y1, x2, y2).
        box2 (tuple): 두 번째 박스의 좌표 (x1, y1, x2, y2).

    Returns:
        bool: 겹치면 True, 그렇지 않으면 False.
    """
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or 
                box1[3] <= box2[1] or box1[1] >= box2[3])