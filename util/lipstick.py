# lipstick.py

import cv2
import numpy as np
from util.detect import get_landmarks, get_lip_points
from util.utils import get_color_from_json

def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1.2):
    dst = src.copy()
    dst = dst / 255.
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

def apply_lipstick(image: np.ndarray, prdCode: str) -> np.ndarray:
    lip_color, option = get_color_from_json(prdCode)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        print("No faces detected.")
        return image

    lip_points = get_lip_points(landmarks)
    
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [lip_points], lip_color)
    
    contour_mask = np.zeros_like(image)
    contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_mask, contours, -1, lip_color, 1) 
    lip_liner = cv2.GaussianBlur(contour_mask, (15, 15), 0)
    
    if option == "Glossy":
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.1, 0.0)
        corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)
        corrected_lips = cv2.GaussianBlur(corrected_lips, (15, 15), 4)
        lip_liner = gamma_correction(lip_liner, 0.8, 1.5)
        corrected_lips = cv2.addWeighted(corrected_lips, 0.8, lip_liner, 0.2, 0.0)
        corrected_lips = cv2.addWeighted(corrected_lips, 1.2, corrected_lips, -0.2, 0)  

    elif option == "Matte":
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)
        corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)
        lip_liner = gamma_correction(lip_liner, 0.8, 1.5)
        corrected_lips = cv2.addWeighted(corrected_lips, 0.7, lip_liner, 0.3, 0)  
        corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 0)
        corrected_lips = cv2.addWeighted(corrected_lips, 1.1, corrected_lips, -0.1, 0)  

    lips_only = np.zeros_like(image)
    lips_only = cv2.fillPoly(lips_only, [lip_points], (255, 255, 255))

    final_image = np.where(lips_only == np.array([255, 255, 255]), corrected_lips, image_with_lipstick)
    return final_image
