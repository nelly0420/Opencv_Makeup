# blush.py

import cv2
import numpy as np
from util.detect import get_landmarks, get_cheek_centers
from util.utils import get_color_from_json

blush_alpha = 0.12  # 블러셔 투명도 (0.0 ~ 1.0)
blush_radius = 45  # 블러셔 적용 반경 (픽셀 단위, 조정 가능)
blush_offset = -36  # 볼 위치를 위로 이동시키는 오프셋 값 (픽셀 단위)
x_offset = 5  # 볼 위치를 왼쪽으로 이동시키는 오프셋 값 (픽셀 단위)

def apply_blush(image, prdCode):
    # 색상 정보를 JSON에서 가져오기
    blush_color, _ = get_color_from_json(prdCode)

    # 이미지의 BGR을 BGRA로 변환 (알파 채널 추가)
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 이미지에서 얼굴 감지
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    # 얼굴 랜드마크 예측
    for face in [landmarks]:  # 현재는 얼굴 하나만 처리
        shape = landmarks

        # 왼쪽과 오른쪽 볼 중심점 계산
        left_cheek_center, right_cheek_center = get_cheek_centers(shape, blush_offset, x_offset)
        
        for cheek_center in [left_cheek_center, right_cheek_center]:
            # 블러셔 영역을 위한 마스크 생성
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.circle(mask, (int(cheek_center[0]), int(cheek_center[1])), blush_radius, (1), -1, cv2.LINE_AA)

            # 마스크를 기반으로 알파 채널 생성
            alpha_channel = (mask * blush_alpha * 255).astype(np.uint8)
            
            # 알파 채널에 Gaussian 블러 적용 (부드러운 블러셔 효과)
            alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            
            # 알파 채널에 미디언 블러 적용 (잡음 제거 및 부드러움 추가)
            alpha_channel = cv2.medianBlur(alpha_channel, 21)

            # 블러셔 색상 이미지 생성
            blush = np.zeros_like(image_bgra, dtype=np.uint8)
            blush[:, :, :3] = blush_color  # 색상 채널
            blush[:, :, 3] = alpha_channel  # 알파 채널
            
            # 알파 채널을 고려하여 최종 이미지를 업데이트
            alpha_mask = alpha_channel / 255.0  # 알파 값을 0과 1 사이로 정규화
            for c in range(0, 3):
                # 각 색상 채널에 대해 블러셔와 원본 이미지 블렌딩
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * blush[:, :, c]

    return image
