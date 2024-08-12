import cv2
import numpy as np
from util.detect import get_landmarks, get_eye_points, create_shifted_points, add_intermediate_points, expand_points
from util.utils import get_color_from_json

eyeshadow_alpha = 0.7  # 아이섀도우의 투명도 설정 (0.0 ~ 1.0)

def apply_eyeshadow(image, prdCode):
    # 색상 정보 얻기
    bgr_color, option1 = get_color_from_json(prdCode)
    print(f"Primary color (BGR): {bgr_color}, Option1 color (Hex): {option1}")

    bgr_color2 = None
    if option1 != "None":
        option1 = option1.lstrip('#')
        rgb_color2 = tuple(int(option1[i:i+2], 16) for i in (0, 2, 4))  # Hex to RGB
        bgr_color2 = (rgb_color2[2], rgb_color2[1], rgb_color2[0])  # RGB to BGR
        print(f"Option1 color (BGR): {bgr_color2}")

    # 이미지 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(image)

    if landmarks is None:
        print("No faces detected.")
        return image

    # 랜드마크 점 추출
    def get_landmark_points(landmarks):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    landmarks = get_landmark_points(landmarks)

    # 랜드마크 36-39 (왼쪽 눈)과 42-45 (오른쪽 눈)를 사용하여 반달 모양 생성
    left_eye_indices = range(36, 40)
    right_eye_indices = range(42, 46)

    left_eye_points = landmarks[left_eye_indices]
    right_eye_points = landmarks[right_eye_indices]

    # 각 점 사이에 중간 점 추가
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    # 랜드마크 확장 (눈 옆에 두 개의 포인트 추가)
    additional_left_points = np.array([
        landmarks[37] + np.array([7, 0]),  # 왼쪽으로 확장
        landmarks[39] + np.array([7, 0])   # 왼쪽으로 확장
    ])
    additional_right_points = np.array([
        landmarks[43] + np.array([7, 0]),  # 오른쪽으로 확장
        landmarks[45] + np.array([7, 0])   # 오른쪽으로 확장
    ])

    # 기존 포인트와 추가 포인트를 병합
    left_eye_points = np.vstack((left_eye_points, additional_left_points))
    right_eye_points = np.vstack((right_eye_points, additional_right_points))

    # 중간 점 추가
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    # 각 점 사이에 중간 점 추가
    left_eye_points = np.array(left_eye_points)
    right_eye_points = np.array(right_eye_points)

    # 점들을 y축으로 이동시키고, 약간 확장
    y_shift1 = 18  # 첫 번째 이동 거리 (조금 작은 값)
    y_shift2 = 26  # 두 번째 이동 거리 (조금 작은 값)
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

        # Alpha 채널의 블러링
        blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
        blurred_alpha_channel = cv2.medianBlur(blurred_alpha_channel, 27)

        # 아이섀도우 생성
        eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
        eyeshadow[:, :, :3] = color
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

            # Alpha 채널의 블러링
            blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (75, 75), 0)
            blurred_alpha_channel = cv2.medianBlur(blurred_alpha_channel, 21)

            # 아이섀도우 생성
            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = color
            eyeshadow[:, :, 3] = blurred_alpha_channel

            # 최종 이미지에 아이섀도우 적용
            alpha_mask = blurred_alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    return image
