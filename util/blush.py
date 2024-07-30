import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화: 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blush_alpha = 0.15  # 블러셔 투명도 (0.0 ~ 1.0)
blush_radius = 35  # 블러셔 적용 반경 (픽셀 단위, 조정 가능)
blush_offset = -20  # 볼 위치를 위로 이동시키는 오프셋 값 (픽셀 단위)
x_offset = 5  # 볼 위치를 왼쪽으로 이동시키는 오프셋 값 (픽셀 단위)

def get_cheek_centers(shape, y_offset, x_offset):
    # 왼쪽과 오른쪽 볼에 해당하는 얼굴 랜드마크 인덱스
    left_cheek_idxs = [1, 2, 3, 4, 31, 49]
    right_cheek_idxs = [12, 13, 14, 15, 35, 53]
    
    # 왼쪽 볼 중심점 계산
    left_cheek_x = np.mean([shape.part(i).x for i in left_cheek_idxs]) + x_offset
    left_cheek_y = np.mean([shape.part(i).y for i in left_cheek_idxs]) + y_offset
    left_cheek_center = np.array([left_cheek_x, left_cheek_y])
    
    # 오른쪽 볼 중심점 계산
    right_cheek_x = np.mean([shape.part(i).x for i in right_cheek_idxs]) + x_offset
    right_cheek_y = np.mean([shape.part(i).y for i in right_cheek_idxs]) + y_offset
    right_cheek_center = np.array([right_cheek_x, right_cheek_y])
    
    return left_cheek_center, right_cheek_center

def apply_blush(image, prdCode):
    # 색상 정보를 JSON에서 가져오기
    blush_color, option = get_color_from_json(prdCode)

    # 이미지의 BGR을 BGRA로 변환 (알파 채널 추가)
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 이미지에서 얼굴 감지
    faces = detector(image, 1)
    if len(faces) == 0:
        print("No faces detected.")
        return image

    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(image, d)
        
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
