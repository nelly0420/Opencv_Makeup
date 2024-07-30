import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eyeshadow_alpha = 0.5  # 투명도 (조정 가능)

def add_intermediate_points(points):
    detailed_points = []
    for i in range(len(points) - 1):
        detailed_points.append(points[i])
        mid_point = ((points[i][0] + points[i + 1][0]) // 2, (points[i][1] + points[i + 1][1]) // 2)
        detailed_points.append(mid_point)
    detailed_points.append(points[-1])
    return np.array(detailed_points, dtype=np.int32)

def transform_to_right_angle_trapezoid(points):
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    n = len(y_coords)
    y_coords_trapezoid = np.copy(y_coords)
    mid_index = n // 2
    for i in range(mid_index):
        y_coords_trapezoid[i] = y_coords[i] - (y_coords[i] - y_coords[-1]) * (i / mid_index)
    transformed_points = np.vstack((x_coords, y_coords_trapezoid)).T
    return transformed_points.astype(np.int32)

def get_eye_brow_midpoints(shape, eye_points, brow_points):
    if len(brow_points) < len(eye_points):
        eye_points = eye_points[:len(brow_points)]
    elif len(brow_points) > len(eye_points):
        brow_points = brow_points[:len(eye_points)]

    midpoints = []
    for i in range(len(eye_points)):
        eye_point = np.array([shape.part(eye_points[i]).x, shape.part(eye_points[i]).y])
        brow_point = np.array([shape.part(brow_points[i]).x, shape.part(brow_points[i]).y])
        midpoint = ((eye_point[0] + brow_point[0]) // 2, (eye_point[1] + brow_point[1]) // 2)
        midpoints.append(midpoint)
    return np.array(midpoints, dtype=np.int32)

def apply_eyeshadow(image, prdCode):
    bgr_color, option1 = get_color_from_json(prdCode)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected.")
        return image

    for face in faces:
        shape = predictor(gray, face)

        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]
        left_brow_points = [17, 18, 19, 20, 21]
        right_brow_points = [22, 23, 24, 25, 26]

        left_midpoints = get_eye_brow_midpoints(shape, left_eye_points, left_brow_points)
        right_midpoints = get_eye_brow_midpoints(shape, right_eye_points, right_brow_points)

        left_eye_points = [(shape.part(point).x, shape.part(point).y) for point in left_eye_points]
        right_eye_points = [(shape.part(point).x, shape.part(point).y) for point in right_eye_points]

        left_eye_points = np.array(left_eye_points + left_midpoints.tolist())
        right_eye_points = np.array(right_eye_points + right_midpoints.tolist())

        left_eye_points = add_intermediate_points(left_eye_points)
        right_eye_points = add_intermediate_points(right_eye_points)

        left_eye_points = transform_to_right_angle_trapezoid(left_eye_points)
        right_eye_points = transform_to_right_angle_trapezoid(right_eye_points)

        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        for eye_points in [left_eye_points, right_eye_points]:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(mask, [eye_points], (1))

            alpha_channel = (mask * eyeshadow_alpha * 255).astype(np.uint8)
            alpha_channel = cv2.GaussianBlur(alpha_channel, (45, 45), 0)
            alpha_channel = cv2.medianBlur(alpha_channel, 21)

            eyeshadow = np.zeros_like(image_bgra, dtype=np.uint8)
            eyeshadow[:, :, :3] = bgr_color
            eyeshadow[:, :, 3] = alpha_channel

            alpha_mask = alpha_channel / 255.0
            for c in range(0, 3):
                image[:, :, c] = (1.0 - alpha_mask) * image[:, :, c] + alpha_mask * eyeshadow[:, :, c]

    return image

