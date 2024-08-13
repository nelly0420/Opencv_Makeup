import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json
from util.detect import get_landmarks, get_eyebrows

# 특정 포인트의 위치 조정
def adjust_point(points, target_index, x_index, y_index):
    
    # 대상 포인트를 복사합니다
    target_point = points[target_index].copy()
    
    # 참조 포인트로부터 새로운 x와 y 좌표를 가져옵니다
    new_x = points[x_index][0]
    new_y = points[y_index][1]
    
    # 대상 포인트의 x와 y 좌표를 업데이트합니다
    target_point[0] = new_x
    target_point[1] = new_y
    
    # 업데이트된 포인트를 배열에 다시 할당합니다
    points[target_index] = target_point
    
    return points


def apply_eyebrow(image, prdCode):
    brow_color, option,_ = get_color_from_json(prdCode)
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        print("No faces detected.")
        return image
    
    # nparray 형태
    left_eyebrow_points, right_eyebrow_points = get_eyebrows(landmarks)

    # move_up = np.array([0, -5])
    # left_eyebrow_points += move_up
    # right_eyebrow_points += move_up

    # 눈썹 앞머리 조정
    left_eyebrow_points = adjust_point(left_eyebrow_points, 3, 4, 2)
    right_eyebrow_points = adjust_point(right_eyebrow_points, 1, 0, 2)

    mask = np.zeros_like(image)

    cv2.fillPoly(mask, [left_eyebrow_points], brow_color)
    cv2.fillPoly(mask, [right_eyebrow_points], brow_color)

    contour_mask = np.zeros_like(image)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(contour_mask, contours, -1, brow_color, 1)
    eyebrow_liner = cv2.GaussianBlur(contour_mask, (21, 21), 0)
    eyebrow_liner = cv2.GaussianBlur(eyebrow_liner, (15, 15), 0)
    
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 4)

    image_with_eyebrows = cv2.addWeighted(image, 1, blurred_mask, 0.5, 0)
    image_with_eyebrows = cv2.addWeighted(image_with_eyebrows, 1, eyebrow_liner, 0.7, 0)
    

    return image_with_eyebrows

