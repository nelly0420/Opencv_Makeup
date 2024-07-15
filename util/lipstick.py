import cv2
import dlib
import numpy as np
from utils import get_color_from_json

upper_lip_idx = list(range(50, 61))
lower_lip_idx = list(range(61, 68)) + list(range(48, 50))

def lip_mask(src: np.ndarray, points: np.ndarray, color):

    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask


def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1):

    dst = src.copy()
    dst = dst / 255.
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst


def apply_lipstick(image: np.ndarray, prdCode):
    color, option1 = get_color_from_json(prdCode)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        upper_lip_points = landmarks[upper_lip_idx]
        lower_lip_points = landmarks[lower_lip_idx]
        
        lip_points = np.concatenate((upper_lip_points, lower_lip_points), axis=0)
        
        mask = lip_mask(image, lip_points, color)
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)

        
       # 입술 영역에 감마 보정 적용
        lips_only = np.zeros_like(image)
        lips_only = cv2.fillPoly(lips_only, [lip_points], (255, 255, 255))
        
        if option1 == "Glossy":
            corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)
            corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 2)
        elif option1 == "Matte":
            corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)


        #corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)
        final_image = np.where(lips_only == np.array([255, 255, 255]), corrected_lips, image_with_lipstick)
        
        return final_image


