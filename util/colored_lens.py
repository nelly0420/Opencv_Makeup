import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    pupils = []
    for face in faces:
        shape = predictor(gray, face)
        left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        # 동공 위치와 반지름 계산
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        left_eye_radius = int(np.linalg.norm(np.array(left_eye_points[1]) - np.array(left_eye_points[4])) / 2.2)  # 살짝 크게 설정

        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        right_eye_radius = int(np.linalg.norm(np.array(right_eye_points[1]) - np.array(right_eye_points[4])) / 2.2)  # 살짝 크게 설정

        pupils.append((left_eye_center, left_eye_radius))
        pupils.append((right_eye_center, right_eye_radius))

    return pupils

def apply_lens(image, prdCode, userColor):
    pupils = detect_pupil(image)

    # 컬러 코드 (HEX -> BGR 변환)
    lens_color = tuple(int(userColor.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

    for (eye_center, eye_radius) in pupils:
        lens_layer = np.zeros_like(image, dtype=np.uint8)

        # 도넛 형태의 렌즈 그리기
        outer_radius = eye_radius
        inner_radius = int(eye_radius * 0.4)  # 도넛의 안쪽 반지름 (40% 정도로 설정)

        # 외부 원 그리기
        cv2.circle(lens_layer, tuple(eye_center), outer_radius, lens_color, -1, cv2.LINE_AA)

        # 내부 원 그리기 (도넛 모양을 만들기 위해)
        cv2.circle(lens_layer, tuple(eye_center), inner_radius, (0, 0, 0), -1, cv2.LINE_AA)

        # 원본 이미지와 렌즈를 적용한 레이어를 합성 (동공 부분에만)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, tuple(eye_center), outer_radius, (1, 1, 1), -1, cv2.LINE_AA)
        cv2.circle(mask, tuple(eye_center), inner_radius, (0, 0, 0), -1, cv2.LINE_AA)

        # 동공 부분만 색칠 (투명도 조절)
        lens_alpha = 0.17  # 투명도를 설정 (0.5로 설정)
        image = cv2.add(image * (1 - mask * lens_alpha), lens_layer * mask * lens_alpha)

    return image
