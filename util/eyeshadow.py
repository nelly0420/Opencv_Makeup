import cv2
import numpy as np
from util.detect import get_landmarks, get_eye_points, create_shifted_points, add_intermediate_points, expand_points
from util.utils import get_color_from_json
eyeshadow_alpha = 0.7  # 아이섀도우의 투명도 설정 (0.0 ~ 1.0)

def get_landmark_points(landmarks):
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

def apply_eyeshadow(image: np.ndarray, prdCode: str, color: str, option2: str) -> np.ndarray:
    # 색상 정보 얻기
    bgr_color, _, option2 = get_color_from_json(prdCode)
    bgr_color2 = None
    if option2 != "None":
        option2 = option2.lstrip('#')
        bgr_color2 = tuple(int(option2[i:i+2], 16) for i in (4, 2, 0))  # Hex to RGB

    # 이미지 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    landmarks = get_landmark_points(landmarks)
    left_eye_indices = range(36, 40)
    right_eye_indices = range(42, 46)
    left_eye_points = landmarks[left_eye_indices]
    right_eye_points = landmarks[right_eye_indices]
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    additional_left_points = np.array([
        landmarks[37] + np.array([7, 0]),
        landmarks[39] + np.array([7, 0])
    ])
    additional_right_points = np.array([
        landmarks[43] + np.array([7, 0]),
        landmarks[45] + np.array([7, 0])
    ])
    left_eye_points = np.vstack((left_eye_points, additional_left_points))
    right_eye_points = np.vstack((right_eye_points, additional_right_points))
    left_eye_points = add_intermediate_points(left_eye_points.tolist(), num_points=2)
    right_eye_points = add_intermediate_points(right_eye_points.tolist(), num_points=2)

    y_shift1 = 18
    y_shift2 = 20
    left_shifted_points1 = create_shifted_points(left_eye_points, y_shift1)
    left_shifted_points2 = create_shifted_points(left_eye_points, y_shift2)
    right_shifted_points1 = create_shifted_points(right_eye_points, y_shift1)
    right_shifted_points2 = create_shifted_points(right_eye_points, y_shift2)
    left_shifted_points1 = expand_points(left_shifted_points1, factor=1.2)
    left_shifted_points2 = expand_points(left_shifted_points2, factor=1.4)
    right_shifted_points1 = expand_points(right_shifted_points1, factor=1.2)
    right_shifted_points2 = expand_points(right_shifted_points2, factor=1.4)

    left_eye_area1 = np.vstack((left_eye_points, left_shifted_points1[::-1]))
    left_eye_area2 = np.vstack((left_shifted_points1, left_shifted_points2[::-1]))
    right_eye_area1 = np.vstack((right_eye_points, right_shifted_points1[::-1]))
    right_eye_area2 = np.vstack((right_shifted_points1, right_shifted_points2[::-1]))

    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    def apply_color_to_area(eye_areas, color):
        for eye_area in eye_areas:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(mask, [eye_area], (1))
            alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)
            blurred_alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            blurred_alpha_channel = cv2.medianBlur(blurred_alpha_channel, 27)
            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = color
            eyeshadow[:, :, 3] = blurred_alpha_channel
            alpha_mask = blurred_alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    apply_color_to_area([left_eye_area1, right_eye_area1], bgr_color)
    
    if bgr_color2:
        apply_color_to_area([left_eye_area2, right_eye_area2], bgr_color2)

    return image