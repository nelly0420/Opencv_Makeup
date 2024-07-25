import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

def add_intermediate_points(points):
    detailed_points = []
    for i in range(len(points) - 1):
        detailed_points.append(points[i])
        # 중간점 추가
        mid_point = ((points[i][0] + points[i + 1][0]) // 2, (points[i][1] + points[i + 1][1]) // 2)
        detailed_points.append(mid_point)
    detailed_points.append(points[-1])


    return np.array(detailed_points, dtype=np.int32)

def transform_to_right_angle_trapezoid(points):
    # y 값을 오른쪽 끝에서 왼쪽 끝으로 선형적으로 감소시키되, 중간에서 원래 값으로 돌아옴
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    n = len(y_coords)
    
    # 선형적으로 감소시키되 끝부분에서 원래 값으로 돌아오게 함
    y_coords_trapezoid = np.copy(y_coords)
    mid_index = n // 2
    for i in range(mid_index):
        y_coords_trapezoid[i] = y_coords[i] - (y_coords[i] - y_coords[-1]) * (i / mid_index)
    
    transformed_points = np.vstack((x_coords, y_coords_trapezoid)).T
    return transformed_points.astype(np.int32)

def apply_eyebrow(image, prdCode):
    bgr_color, option1 = get_color_from_json(prdCode)

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()    
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        
        left_eyebrow_points = []
        for i in range(17, 22): 
            x = shape.part(i).x
            y = shape.part(i).y
            left_eyebrow_points.append((x, y))

        right_eyebrow_points = []
        for i in range(22, 27):
            x = shape.part(i).x
            y = shape.part(i).y
            right_eyebrow_points.append((x, y))

        right_eyebrow_points = np.array(right_eyebrow_points)
        left_eyebrow_points = np.array(left_eyebrow_points)

        move_up = np.array([0, -5])
        left_eyebrow_points += move_up
        right_eyebrow_points += move_up

        left_eyebrow_points = add_intermediate_points(left_eyebrow_points)
        cv2.fillPoly(image, [right_eyebrow_points], bgr_color)
        right_eyebrow_points = add_intermediate_points(right_eyebrow_points)
        right_eyebrow_points = transform_to_right_angle_trapezoid(right_eyebrow_points)

        cv2.fillPoly(image, [left_eyebrow_points], bgr_color)

        cv2.fillPoly(image, [right_eyebrow_points], bgr_color)
        
    return image


