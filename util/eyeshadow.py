import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화
detector = dlib.get_frontal_face_detector()  # 얼굴 감지기 초기화
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 얼굴 랜드마크 예측기 초기화

eyeshadow_alpha = 0.3  # 아이섀도우의 투명도 설정 (0.0 ~ 1.0)

def get_midpoint(point1, point2):
    """
    두 점 사이의 중간 점을 계산합니다.
    """
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

def add_intermediate_points(points):
    """
    각 점 사이에 중간 점을 추가하여 점의 수를 증가시킵니다.
    """
    detailed_points = []
    for i in range(len(points) - 1):
        detailed_points.append(points[i])
        mid_point = ((points[i][0] + points[i + 1][0]) // 2, (points[i][1] + points[i + 1][1]) // 2)
        detailed_points.append(mid_point)
    detailed_points.append(points[-1])
    return np.array(detailed_points, dtype=np.int32)

def apply_eyeshadow(image, prdCode):
    """
    이미지에 아이섀도우를 적용합니다.
    """
    # 색상 정보 얻기
    bgr_color, option1 = get_color_from_json(prdCode)

    # 이미지 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # 얼굴 감지

    if len(faces) == 0:
        print("No faces detected.")
        return image

    for face in faces:
        shape = predictor(gray, face)  # 얼굴 랜드마크 예측

        # 눈과 눈썹의 랜드마크 좌표 추출
        left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        left_brow_points = [(shape.part(i).x, shape.part(i).y) for i in range(17, 22)]
        right_brow_points = [(shape.part(i).x, shape.part(i).y) for i in range(22, 27)]

        left_eye_points = np.array(left_eye_points)
        right_eye_points = np.array(right_eye_points)
        left_brow_points = np.array(left_brow_points)
        right_brow_points = np.array(right_brow_points)

        # 눈과 눈썹 사이의 중간 점 계산
        left_midpoints = [get_midpoint(left_eye_points[i % len(left_eye_points)], left_brow_points[i % len(left_brow_points)]) for i in range(len(left_eye_points))]
        right_midpoints = [get_midpoint(right_eye_points[i % len(right_eye_points)], right_brow_points[i % len(right_brow_points)]) for i in range(len(right_eye_points))]

        # 눈과 중간 점을 연결하여 아이섀도우 영역 생성
        left_eye_area = np.vstack((left_eye_points, left_midpoints[::-1]))
        right_eye_area = np.vstack((right_eye_points, right_midpoints[::-1]))

        # BGRA 형식으로 이미지 변환 (Alpha 채널 추가)
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        for eye_area, eye_points in zip([left_eye_area, right_eye_area], [left_eye_points, right_eye_points]):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(mask, [eye_area], (1))  # 아이섀도우 적용할 영역을 마스크로 생성

            # 동공 영역 제외 (눈동자)
            eye_hull = cv2.convexHull(eye_points)
            cv2.fillPoly(mask, [eye_hull], (0))

            # Alpha 채널 생성
            alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)

            # 그라데이션 효과 추가
            for i in range(len(eye_points) - 1):
                x1, y1 = eye_points[i]
                x2, y2 = eye_points[i + 1]
                gradient_strength = (i / len(eye_points)) * 0.5 + 0.5  # 그라데이션 강도 설정
                cv2.line(alpha_channel, (x1, y1), (x2, y2), (255 * gradient_strength), 2)

            # Alpha 채널의 블러링
            alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            alpha_channel = cv2.medianBlur(alpha_channel, 21)

            # 아이섀도우 생성
            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = bgr_color
            eyeshadow[:, :, 3] = alpha_channel

            # 최종 이미지에 아이섀도우 적용
            alpha_mask = alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    return image
