import cv2
import dlib
import numpy as np
from collections import OrderedDict

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 볼 부분 랜드마크 인덱스 정의
CHEEK_IDXS = OrderedDict([
    ("left_cheek", [1, 2, 3, 4, 49, 31]),
    ("right_cheek", [ 12, 13, 14, 15, 35, 53])
])

# 블러쉬 색상 및 투명도
blush_color = (193, 153, 255)  # 연한 분홍색 계열 (BGR 포맷)
blush_alpha = 0.3  # 투명도

def apply_blush(image):
    # 이미지 BGRA로 변환
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    faces = detector(image, 1)
    if len(faces) == 0:
        print("No faces detected.")
        return image

    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(image, d)
        
        for (_, name) in enumerate(CHEEK_IDXS.keys()):
            pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32)
            for i, j in enumerate(CHEEK_IDXS[name]):
                pts[i] = [shape.part(j).x, shape.part(j).y]

            # 볼 영역 마스크 생성
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

            # 블러쉬 컬러 채널 생성 및 적용
            blush = np.zeros_like(image_bgra)
            blush[:, :, :3] = blush_color  # BGR 포맷
            blush[:, :, 3] = (mask * blush_alpha * 255).astype(np.uint8)  # 알파 채널 설정

            # 가우시안 블러 적용
            blush[:, :, 3] = cv2.GaussianBlur(blush[:, :, 3], (25, 25), 0)  # 커널 크기 조정
            # 추가된 미디안 블러 적용
            blush[:, :, 3] = cv2.medianBlur(blush[:, :, 3], 7)  # 커널 크기 조정

            # 알파 채널 고려하여 최종 합성
            alpha_mask = blush[:, :, 3] / 255.0
            for c in range(0, 3):
                image[:, :, c] = image[:, :, c] * (1 - alpha_mask) + blush[:, :, c] * alpha_mask

    return image