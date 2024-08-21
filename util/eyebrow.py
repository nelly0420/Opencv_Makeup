import cv2
import numpy as np
from util.utils import get_color_from_json
from util.detect import get_landmarks, get_eyebrows

def adjust_point(points, target_index, x_index, y_index):
    """Adjust a specific point's location based on other points."""
    target_point = points[target_index].copy()
    new_x = points[x_index][0]
    new_y = points[y_index][1]

    target_point[0] = new_x
    target_point[1] = new_y

    points[target_index] = target_point

    return points

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert HEX color code to BGR color format."""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))  # Skip the '#'
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR

def apply_eyebrow2(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """
    Apply eyebrows to the image using a specified product code and color.
    """

    brow_color, _, _ = get_color_from_json(prdCode)

    # 사용자 정의 색상
    brow_color = hex_to_bgr(color) if color is not None else get_color_from_json(prdCode)[0]

    # Detect facial landmarks
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    # Get eyebrow points
    left_eyebrow_points, right_eyebrow_points = get_eyebrows(landmarks)
    
    left_eyebrow_points = adjust_point(left_eyebrow_points, 3, 4, 2)
    right_eyebrow_points = adjust_point(right_eyebrow_points, 1, 0, 2)

    # 눈썹 영역의 마스크 생성
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [left_eyebrow_points, right_eyebrow_points], brow_color)

    # brow 윤곽선 영역
    contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_mask = np.zeros_like(image)
    cv2.drawContours(contour_mask, contours, -1, brow_color, 1)
    blurred_browliner = cv2.GaussianBlur(contour_mask, (19,19), 0)

    # 마스크 영역에서 스케치 효과 적용
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 5, 50)

    # 스케치 효과 주기
    ret, sketch = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY_INV)
    sketch_colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    #blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 원본 이미지에 눈썹 영역 합성
    image_with_sketch_eyebrows = cv2.bitwise_and(sketch_colored, mask)
    blurred_mask = cv2.GaussianBlur(image_with_sketch_eyebrows, (5, 5), 0)
    eyebrow_mask = cv2.addWeighted(blurred_browliner, 0.3, blurred_mask, 0.5, 0)
    result = cv2.addWeighted(image, 1, eyebrow_mask, 0.3, 1)
    
    return result