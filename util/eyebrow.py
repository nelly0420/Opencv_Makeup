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

# def apply_eyebrow2(image, prdCode, color):
#     """
#     Apply eyebrows to the image using a specified product code and color.
#     Plus, 점이 아닌 색상 차이를 이용
#     """

#     brow_color, _, _ = get_color_from_json(prdCode)

#     # 사용자 정의 색상
#     userColor = hex_to_bgr(color)
#     if color is not None:
#         brow_color = userColor

#     # Detect facial landmarks
#     landmarks = get_landmarks(image)
#     if landmarks is None:
#         print("No faces detected.")
#         return image

#     # Get eyebrow points
#     left_eyebrow_points, right_eyebrow_points = get_eyebrows(landmarks)

#     # y 값의 min과 max 추출
#     eyebrow_y_min = min(np.min(left_eyebrow_points[:, 1]), np.min(right_eyebrow_points[:, 1])) - 10  # y 범위 상향 조정
#     eyebrow_y_max = max(np.max(left_eyebrow_points[:, 1]), np.max(right_eyebrow_points[:, 1]))

#     # x 값의 min과 max 추출 (눈썹의 좌우 영역 한정)
#     eyebrow_x_min = min(np.min(left_eyebrow_points[:, 0]), np.min(right_eyebrow_points[:, 0]))
#     eyebrow_x_max = max(np.max(left_eyebrow_points[:, 0]), np.max(right_eyebrow_points[:, 0]))


    # # 이미지를 HSV 색상 공간으로 변환
    # hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # 눈썹의 HSV 범위 설정
    # lower_eyebrow = np.array([0, 20, 30])  # 어두운 색상 (검정색, 짙은 채도, 어두운 밝기)
    # upper_eyebrow = np.array([30, 255, 90])  # 갈색 계열 (짙은 색상, 중간 밝기)

    # # 눈썹 영역 마스크 생성
    # eyebrow_mask = cv2.inRange(hsv_frame, lower_eyebrow, upper_eyebrow)

    # # 마스크에서 y 값이 eyebrow_y_min과 eyebrow_y_max, x 값이 eyebrow_x_min과 eyebrow_x_max 사이에만 적용
    # mask_applied = np.zeros_like(eyebrow_mask)
    # mask_applied[eyebrow_y_min:eyebrow_y_max, eyebrow_x_min:eyebrow_x_max] = eyebrow_mask[eyebrow_y_min:eyebrow_y_max, eyebrow_x_min:eyebrow_x_max]

    # # 노이즈 제거 (필터링 및 후처리)
    # kernel = np.ones((3, 3), np.uint8)
    # mask_applied = cv2.morphologyEx(mask_applied, cv2.MORPH_CLOSE, kernel)
    # mask_applied = cv2.morphologyEx(mask_applied, cv2.MORPH_OPEN, kernel)

    # # 흰색(255, 255, 255)으로 채울 이미지 생성
    # white_filled_image = np.zeros_like(image)
    # white_filled_image[mask_applied > 0] = [255, 255, 255]

    # # 원래 이미지 색상은 유지하고, 눈썹 영역만 흰색으로 표시
    # final_image = np.where(white_filled_image == np.array([255, 255, 255]), white_filled_image, image)

    # return final_image

    # ------------------------------------------> 흑백처리
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # 눈썹 영역 내의 검은 부분을 감지하기 위해 이미지를 흑백(Grayscale)으로 변환
    # roi_gray = gray[eyebrow_y_min:eyebrow_y_max, eyebrow_x_min:eyebrow_x_max]

    # # 눈썹 영역의 검은 부분 마스크 생성 (임계값을 사용하여 검은 부분을 감지)
    # _, eyebrow_mask = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)

    # # 흰색(255, 255, 255)으로 채울 이미지 생성
    # white_filled_image = np.zeros_like(image)
    # white_filled_image[eyebrow_y_min:eyebrow_y_max, eyebrow_x_min:eyebrow_x_max][eyebrow_mask > 0] = [255, 255, 255]

    # # 원래 이미지 색상은 유지하고, 눈썹 영역만 흰색으로 표시
    # final_image = np.where(white_filled_image == np.array([255, 255, 255]), white_filled_image, image)

    # return final_image



#-------------------> fillPoly
def apply_eyebrow2(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """
    Apply eyebrows to the image using a specified product code and color.
    """

    brow_color, _, _ = get_color_from_json(prdCode)

    # 사용자 정의 색상
    userColor = hex_to_bgr(color)
    if color is not None:
        brow_color = userColor

    # Detect facial landmarks
    landmarks = get_landmarks(image)
    if landmarks is None:
        print("No faces detected.")
        return image

    # Get eyebrow points
    left_eyebrow_points, right_eyebrow_points = get_eyebrows(landmarks)
    
    left_eyebrow_points = adjust_point(left_eyebrow_points, 3, 4, 2)
    right_eyebrow_points = adjust_point(right_eyebrow_points, 1, 0, 2)

    # # Adjust the points by shifting the y-coordinate by +1
    # shifted_left_eyebrow_points = np.array([[x, y + 2] for x, y in left_eyebrow_points])
    # shifted_right_eyebrow_points = np.array([[x, y + 2] for x, y in right_eyebrow_points])

    # # Combine the original and shifted points for fillPoly
    # combined_left_eyebrow = np.concatenate((left_eyebrow_points, shifted_left_eyebrow_points[::-1]), axis=0)
    # combined_right_eyebrow = np.concatenate((right_eyebrow_points, shifted_right_eyebrow_points[::-1]), axis=0)

    # # Create a mask for the eyebrows
    # mask = np.zeros_like(image, dtype=np.uint8)
    # cv2.fillPoly(mask, [combined_left_eyebrow], brow_color)
    # cv2.fillPoly(mask, [combined_right_eyebrow], brow_color)

    # # Combine the original image with the eyebrow mask
    # image_with_eyebrows = cv2.addWeighted(image, 1, mask, 0.7, 0)

    # return image_with_eyebrows

    # 빨간 점으로 눈썹 포인트 표시
    # for point in left_eyebrow_points:
    #     cv2.circle(image, tuple(point), 2, brow_color, -1)  # 빨간색 점

    # for point in right_eyebrow_points:
    #     cv2.circle(image, tuple(point), 2, brow_color, -1)  # 빨간색 점

    # return image

    #plus) 점간격더좁게 조정(x간의 간격)

    # 눈썹 영역의 폴리곤 마스크 생성
    mask = np.zeros_like(image, dtype=np.uint8)

    cv2.fillPoly(mask, [left_eyebrow_points], brow_color)
    cv2.fillPoly(mask, [right_eyebrow_points], brow_color)

    # 마스크 영역에서 스케치 효과 적용
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 10, 70)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    # 스케치 결과를 컬러 이미지로 변환
    sketch_colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    # 블러 효과 적용
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # 원본 이미지에 스케치 적용된 눈썹 영역 합성
    image_with_sketch_eyebrows = cv2.bitwise_and(sketch_colored, blurred_mask)
    result = cv2.addWeighted(image, 1, image_with_sketch_eyebrows, 0.1, 0)
    
    return result