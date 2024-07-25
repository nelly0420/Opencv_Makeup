import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json


def lip_mask(src: np.ndarray, landmarks, color):
    points = []
    for i in range(landmarks.num_parts):
        points.append((landmarks.part(i).x, landmarks.part(i).y))

    # 윗입술 포인트
    indices_top = [64, 63, 62, 61, 60, 48, 49, 50, 51, 52, 53, 54]
    points_top = [points[i] for i in indices_top]

    # 아랫입술 포인트
    indices_bottom = [54, 55, 56, 57, 58, 59, 48, 60, 67, 66, 65, 64]
    points_bottom = [points[i] for i in indices_bottom]

    # 윗입술과 아랫입술 포인트 결합
    lip_points = np.array(points_top + points_bottom)

    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [lip_points], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask, lip_points

def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1):
    dst = src.copy()
    dst = dst / 255.
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

def apply_lipstick(image, prdCode):
    color, option = get_color_from_json(prdCode)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        
        mask, lip_points = lip_mask(image, shape, color)
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)

        lips_only = np.zeros_like(image)
        lips_only = cv2.fillPoly(lips_only, [lip_points], (255, 255, 255))
        
        if option == "Glossy":
            corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)
            corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 2)
        elif option == "Matte":
            corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)
        
        final_image = np.where(lips_only == np.array([255, 255, 255]), corrected_lips, image_with_lipstick)
        
        return final_image
