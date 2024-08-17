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

def apply_eyebrow(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """
    Apply eyebrows to the image using a specified product code and color.

    """

    brow_color, _, _ = get_color_from_json(prdCode)

    # 사용자 정의 색상
    userColor = hex_to_bgr(color)
    if color != None:
        brow_color = userColor

    # Detect facial landmarks
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    # Get eyebrow points
    left_eyebrow_points, right_eyebrow_points = get_eyebrows(landmarks)

    # Adjust eyebrow points if needed
    # left_eyebrow_points = adjust_point(left_eyebrow_points, 3, 4, 2)
    # right_eyebrow_points = adjust_point(right_eyebrow_points, 1, 0, 2)

    # Create a mask for the eyebrows
    mask = np.zeros_like(image, dtype=np.uint8)
    #cv2.fillPoly(mask, [left_eyebrow_points], brow_color)
    cv2.polylines(mask, [left_eyebrow_points], isClosed=False, color = brow_color, thickness=3)
    #cv2.fillPoly(mask, [right_eyebrow_points], brow_color)
    cv2.polylines(mask, [right_eyebrow_points], isClosed=False, color=brow_color, thickness=3)

     # Create a contour mask and apply Gaussian blur
    contour_mask = np.zeros_like(image, dtype=np.uint8)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_mask, contours, -1, brow_color, 1)
    eyebrow_liner = cv2.GaussianBlur(contour_mask, (9, 9), 2)
    #eyebrow_liner = cv2.GaussianBlur(eyebrow_liner, (15, 15), 0)

    # Blur the mask for a smoother effect
    blurred_mask = cv2.GaussianBlur(mask, (11, 11), 2)

    # Combine the original image with the blurred eyebrow mask
    image_with_eyebrows = cv2.addWeighted(image, 1, blurred_mask, 0.7, 0)
    image_with_eyebrows = cv2.addWeighted(image_with_eyebrows, 1, eyebrow_liner, 0.7, 0)

    return image_with_eyebrows
