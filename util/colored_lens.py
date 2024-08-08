import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화: 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

lens_alpha = 0.3  # 렌즈 투명도 (0.0 ~ 1.0)
pupil_radius_ratio = 0.3  # 동공 반지름의 비율 (눈의 너비에 대한 비율)
lens_radius_ratio = 0.7  # 렌즈 반지름의 비율 (눈의 너비에 대한 비율)
blur_radius = 15  # Gaussian 블러를 위한 반경

def get_eye_centers_and_sizes(shape):
    # 왼쪽과 오른쪽 눈에 해당하는 얼굴 랜드마크 인덱스
    left_eye_idxs = [36, 37, 38, 39, 40, 41]
    right_eye_idxs = [42, 43, 44, 45, 46, 47]
    
    # 왼쪽 눈 중심점 및 크기 계산
    left_eye_points = np.array([[shape.part(i).x, shape.part(i).y] for i in left_eye_idxs])
    left_eye_center = left_eye_points.mean(axis=0).astype(int)
    left_eye_width = np.linalg.norm(left_eye_points[0] - left_eye_points[3])

    # 오른쪽 눈 중심점 및 크기 계산
    right_eye_points = np.array([[shape.part(i).x, shape.part(i).y] for i in right_eye_idxs])
    right_eye_center = right_eye_points.mean(axis=0).astype(int)
    right_eye_width = np.linalg.norm(right_eye_points[0] - right_eye_points[3])

    return (left_eye_center, left_eye_width), (right_eye_center, right_eye_width)

def apply_color_lens(image, prdCode):
    # 색상 정보를 JSON에서 가져오기
    lens_color, option = get_color_from_json(prdCode)
    
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
        
        # 왼쪽과 오른쪽 눈 중심점 및 크기 계산
        (left_eye_center, left_eye_width), (right_eye_center, right_eye_width) = get_eye_centers_and_sizes(shape)
        
        for eye_center, eye_width in [(left_eye_center, left_eye_width), (right_eye_center, right_eye_width)]:
            # 동공 및 렌즈 반지름 계산
            pupil_radius = int(eye_width * pupil_radius_ratio)
            lens_radius = int(eye_width * lens_radius_ratio)
            
            # 렌즈 영역을 위한 마스크 생성
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.circle(mask, (int(eye_center[0]), int(eye_center[1])), lens_radius, (1), -1, cv2.LINE_AA)
            
            # 동공 영역 마스크 생성 (원래 눈 색상을 유지)
            pupil_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.circle(pupil_mask, (int(eye_center[0]), int(eye_center[1])), pupil_radius, (1), -1, cv2.LINE_AA)

            # 동공 영역을 제외한 렌즈 영역 마스크
            lens_mask = mask - pupil_mask
            
            # 렌즈 영역의 알파 채널 생성
            alpha_channel = (lens_mask * lens_alpha * 255).astype(np.uint8)
            
            # 알파 채널에 Gaussian 블러 적용 (부드러운 렌즈 효과)
            alpha_channel = cv2.GaussianBlur(alpha_channel, (blur_radius, blur_radius), 0)
            
            # 렌즈 색상 이미지 생성
            lens = np.zeros_like(image_bgra, dtype=np.uint8)
            lens[:, :, :3] = lens_color  # 색상 채널
            lens[:, :, 3] = alpha_channel  # 알파 채널
            
            # 알파 채널을 고려하여 최종 이미지를 업데이트
            alpha_mask = alpha_channel / 255.0  # 알파 값을 0과 1 사이로 정규화
            for c in range(0, 3):
                # 각 색상 채널에 대해 렌즈와 원본 이미지 블렌딩
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * lens[:, :, c]

    return image

