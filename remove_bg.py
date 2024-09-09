import os
import cv2
import numpy as np

def process_image(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 이미지가 흑백인지 컬러인지 확인
    is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
    
    if is_grayscale:
        # 흑백 이미지 처리
        # 알파 채널 추가
        alpha = np.where(img > 0, 255, 0).astype(np.uint8)
        img = cv2.merge([img, img, img, alpha])
        
        # 색 반전 (배경 제외)
        img[:,:,:3] = np.where(img[:,:,3:] > 0, 255 - img[:,:,:3], img[:,:,:3])
    else:
        # 컬러 이미지 처리
        # 검은색 픽셀 찾기 (RGB 모두 0인 경우)
        black_pixels = np.all(img[:,:,:3] == 0, axis=2)
        
        # 알파 채널 생성
        alpha = np.where(black_pixels, 0, 255).astype(np.uint8)
        
        # 알파 채널 추가
        if img.shape[2] == 3:  # 알파 채널이 없는 경우
            img = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])
        else:  # 이미 알파 채널이 있는 경우
            img[:,:,3] = alpha
    
    return img

# 이미지 처리 및 저장
def process_and_save(input_path, output_path):
    processed_img = process_image(input_path)
    cv2.imwrite(output_path, processed_img)
    print(f"처리된 이미지가 {output_path}에 저장되었습니다.")

# 사용 예시
# process_and_save('input_image.png', 'output_image.png')

if __name__ == "__main__":
    import glob

    imgs = glob.glob(os.path.join('demo','base_birds', '*.png'))

    for img in imgs:
        process_and_save(img, os.path.join('demo', 'base_birds_alpha', os.path.basename(img)))
