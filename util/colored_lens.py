import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화
detector = dlib.get_frontal_face_detector()  # 얼굴 감지기 초기화
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 얼굴 랜드마크 예측기 초기화

eyeshadow_alpha = 0.7  # 아이섀도우의 투명도 설정 (0.0 ~ 1.0)

def create_shifted_points(points, y_shift):
    """
    주어진 점들을 y축으로 일정량 이동시킨 새로운 점들을 생성합니다.
    """
    return [(x, y - y_shift) for (x, y) in points]

def add_intermediate_points(points, num_points=1):
    """
    각 점 사이에 지정된 수의 중간 점을 추가하여 점의 수를 증가시킵니다.
    """
    detailed_points = []
    for i in range(len(points) - 1):
        detailed_points.append(points[i])
        for j in range(1, num_points + 1):
            mid_point = (
                points[i][0] + (points[i + 1][0] - points[i][0]) * j // (num_points + 1),
                points[i][1] + (points[i + 1][1] - points[i][1]) * j // (num_points + 1),
            )
            detailed_points.append(mid_point)
    detailed_points.append(points[-1])
    return np.array(detailed_points, dtype=np.int32)

def expand_points(points, factor=1.2):
    """
    주어진 점들을 중심으로 일정 비율로 확장시킨 새로운 점들을 생성합니다.
    """
    center_x = np.mean([x for (x, y) in points])
    center_y = np.mean([y for (x, y) in points])
    expanded_points = [(int(center_x + (x - center_x) * factor), int(center_y + (y - center_y) * factor)) for (x, y) in points]
    return expanded_points

def apply_eyeshadow(image, prdCode):
    """
    이미지에 아이섀도우를 적용합니다.
    """
    # 색상 정보 얻기
    bgr_color, option1 = get_color_from_json(prdCode)
    print(f"Primary color (BGR): {bgr_color}, Option1 color (Hex): {option1}")  # 디버깅 출력문 추가
    
    bgr_color2 = None
    if option1 != "None":
        option1 = option1.lstrip('#')
        rgb_color2 = tuple(int(option1[i:i+2], 16) for i in (0, 2, 4))  # Hex to RGB
        bgr_color2 = (rgb_color2[2], rgb_color2[1], rgb_color2[0])  # RGB to BGR
        print(f"Option1 color (BGR): {bgr_color2}")  # 디버깅 출력문 추가
    
    # 이미지 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # 얼굴 감지

    if len(faces) == 0:
        print("No faces detected.")
        return image

    for face in faces:
        shape = predictor(gray, face)  # 얼굴 랜드마크 예측

        # 랜드마크 36-39 (왼쪽 눈)과 42-45 (오른쪽 눈)를 사용하여 반달 모양 생성
        left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 40)]
        right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 46)]

        # 각 점 사이에 중간 점 추가
        left_eye_points = add_intermediate_points(left_eye_points, num_points=2)
        right_eye_points = add_intermediate_points(right_eye_points, num_points=2)

        # 점들을 y축으로 이동시키고, 약간 확장
        y_shift1 = 15  # 첫 번째 이동 거리
        y_shift2 = 30  # 두 번째 이동 거리
        left_shifted_points1 = create_shifted_points(left_eye_points, y_shift1)
        left_shifted_points2 = create_shifted_points(left_eye_points, y_shift2)
        right_shifted_points1 = create_shifted_points(right_eye_points, y_shift1)
        right_shifted_points2 = create_shifted_points(right_eye_points, y_shift2)

        left_shifted_points1 = expand_points(left_shifted_points1, factor=1.2)
        left_shifted_points2 = expand_points(left_shifted_points2, factor=1.4)
        right_shifted_points1 = expand_points(right_shifted_points1, factor=1.2)
        right_shifted_points2 = expand_points(right_shifted_points2, factor=1.4)

        # 왼쪽 및 오른쪽 눈에 대해 첫 번째 및 두 번째 색상 영역을 생성
        left_eye_area1 = np.vstack((left_eye_points, left_shifted_points1[::-1]))
        left_eye_area2 = np.vstack((left_shifted_points1, left_shifted_points2[::-1]))
        right_eye_area1 = np.vstack((right_eye_points, right_shifted_points1[::-1]))
        right_eye_area2 = np.vstack((right_shifted_points1, right_shifted_points2[::-1]))

        # BGRA 형식으로 이미지 변환 (Alpha 채널 추가)
        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # 첫 번째 색상 적용
        for eye_area, color in zip([left_eye_area1, right_eye_area1], [bgr_color, bgr_color]):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(mask, [eye_area], (1))  # 아이섀도우 적용할 영역을 마스크로 생성

            # Alpha 채널 생성
            alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)

            # 아이섀도우 생성
            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = color
            eyeshadow[:, :, 3] = alpha_channel

            # Alpha 채널의 블러링 추가
            blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            eyeshadow[:, :, 3] = blurred_alpha_channel

            # 최종 이미지에 아이섀도우 적용
            alpha_mask = blurred_alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

        # 두 번째 색상 적용 (option1이 있는 경우)
        if bgr_color2:
            for eye_area, color in zip([left_eye_area2, right_eye_area2], [bgr_color2, bgr_color2]):
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                cv2.fillPoly(mask, [eye_area], (1))  # 아이섀도우 적용할 영역을 마스크로 생성

                # Alpha 채널 생성
                alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)

                # 아이섀도우 생성
                eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
                eyeshadow[:, :, :3] = color
                eyeshadow[:, :, 3] = alpha_channel

                # Alpha 채널의 블러링 추가
                blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
                eyeshadow[:, :, 3] = blurred_alpha_channel

                # 최종 이미지에 아이섀도우 적용
                alpha_mask = blurred_alpha_channel / 255.0
                for c in range(0, 3):
                    image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    return image
