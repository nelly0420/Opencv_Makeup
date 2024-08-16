import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

def apply_sunglasses(image, prdCode):
    color, option, _ = get_color_from_json(prdCode)
    finalimg = image.copy()
    
    # Load dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Load and process the sunglasses filter
    sunglasses_image = cv2.imread("sunglasses.jpg")  # Replace with your actual sunglasses image path
    if sunglasses_image is None:
        raise ValueError("Error loading sunglasses image.")
    
    sunglasses_image = cv2.cvtColor(sunglasses_image, cv2.COLOR_BGR2GRAY)
    sunglasses_image = cv2.cvtColor(sunglasses_image, cv2.COLOR_GRAY2BGR)
    sunglasses_image[np.where((sunglasses_image == [0,0,0]).all(axis=2))] = color
    
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        
        # Get the coordinates for the eyes (landmarks 36-41 and 42-47)
        eye_points_left = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        eye_points_right = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Get bounding boxes for both eyes
        x_left, y_left, x_right, y_right = (float('inf'), float('inf'), float('-inf'), float('-inf'))
        for (x, y) in eye_points_left + eye_points_right:
            x_left = min(x_left, x)
            y_left = min(y_left, y)
            x_right = max(x_right, x)
            y_right = max(y_right, y)
        
        eye_w = x_right - x_left
        eye_h = y_right - y_left
        
        # Resize the sunglasses image to fit the eye region
        sunglasses_resized = cv2.resize(sunglasses_image, (int(eye_w * 1.5), int(eye_h * 1.5)))
        
        # Overlay the sunglasses image onto the image
        for i in range(sunglasses_resized.shape[0]):
            for j in range(sunglasses_resized.shape[1]):
                if np.any(sunglasses_resized[i, j] != [0, 0, 0]):  # Skip black background
                    y_offset = int(y_left + i - 20)
                    x_offset = int(x_left + j - 20)
                    
                    if 0 <= y_offset < finalimg.shape[0] and 0 <= x_offset < finalimg.shape[1]:
                        finalimg[y_offset, x_offset] = sunglasses_resized[i, j]
    
    return finalimg

