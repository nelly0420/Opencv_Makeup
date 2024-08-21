# lipstick.py

import cv2
import numpy as np
from PIL import ImageColor
from util.detect import get_landmarks, get_lip_points
from util.utils import get_color_from_json

def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1.2):
    dst = src.copy()
    dst = dst / 255.
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

def apply_lipstick(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """주어진 색상과 옵션에 따라 립스틱 효과를 적용하는 함수"""
    
    usercolor = ImageColor.getrgb(color) #사용자 정의 color
    user_bgr = (usercolor[2], usercolor[1], usercolor[0])  # Convert RGB to BGR
    
    # Extract the predefined lipstick color and other options from the JSON
    lip_color, option, _ = get_color_from_json(prdCode)

    # Get facial landmarks
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        print("No faces detected.")
        return image

    # Get lip points from landmarks
    lip_points = get_lip_points(landmarks)
    
    # 사용자 정의 컬러가 있으면 립 색상 변경
    if user_bgr != None:
        lip_color = user_bgr

    # Create masks
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, [lip_points], lip_color)  # Apply the selected color to the lip region
    
    contour_mask = np.zeros_like(image)
    contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_mask, contours, -1, lip_color, 1) 
    lip_liner = cv2.GaussianBlur(contour_mask, (19, 19), 1)
    
    corrected_lips = image_with_lipstick = np.copy(image)

    if option == "Glossy":
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.1, 0)
        corrected_lips = gamma_correction(image_with_lipstick, 1.2)
        corrected_lips = cv2.GaussianBlur(corrected_lips, (15, 15), 0)
        lip_liner = gamma_correction(lip_liner, 0.8, 1.5)
        corrected_lips = cv2.addWeighted(corrected_lips, 0.8, lip_liner, 0.2, 0)
        corrected_lips = cv2.addWeighted(corrected_lips, 1.5, corrected_lips, -0.5, 0)  
    
    elif option == "Matte":
        image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.3, 0)
        corrected_lips = gamma_correction(image_with_lipstick, 1.5)
        corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 0)
        #lip_liner = gamma_correction(lip_liner, 0.8, 1.0)
        corrected_lips = cv2.addWeighted(corrected_lips, 0.7, lip_liner, 0.3, 0)  
        corrected_lips = cv2.addWeighted(corrected_lips, 1.5, corrected_lips, -0.5, 0)  

    lips_only = np.zeros_like(image)
    lips_only = cv2.fillPoly(lips_only, [lip_points], (255, 255, 255))

    final_image = np.where(lips_only == np.array([255, 255, 255]), corrected_lips, image_with_lipstick)
    return final_image

def apply_lipstick2(detector, predictor, img, prdCode):
    """
    Applies lipstick to the face in the given image.

    Parameters:
    - img: The input image (numpy array).
    - prdCode: The code for the lipstick color.

    Returns:
    - The image with lipstick applied (numpy array).
    """
    lip_color, option = get_color_from_json(prdCode)

    # 오류가 발생할 경우 기본적으로 원본 이미지를 반환하도록 설정합니다.
    img_with_lipstick = img.copy()
    
    try:
        # Convert image to RGB (dlib expects RGB images)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert image to RGB (if it's RGBA, remove alpha channel)
        # if img.shape[2] == 4:  # RGBA
        #     rgb_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # else:
        #     rgb_img = img  # RGB or grayscale

        # Detect faces in the image
        faces = detector(rgb_img)

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(rgb_img, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

            # Define the lip area using landmark points (landmarks 48 to 67)
            lip_indices = list(range(48, 68))
            lips = [landmarks[i] for i in lip_indices]

            # Create a mask for the lip region
            mask = np.zeros_like(img[:, :, 0])
            lips_points = np.array(lips, np.int32)
            lips_points = lips_points.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [lips_points], 255)

            # 이미지가 3차원 배열인지를 확인합니다. 3차원 배열은 일반적으로 (높이, 너비, 채널 수)의 형태를 가집니다.
            # img.shape[2]는 이미지의 채널 수를 나타냅니다 (예: RGB 이미지의 경우 3, RGBA 이미지의 경우 4).
            num_channels = img.shape[2] if len(img.shape) == 3 else 1

            # Convert HEX color to RGB
            # print(lip_color)
            blue, green, red = lip_color

            # Create the lipstick color (red) with the same number of channels as the image
            lipstick_color = np.zeros_like(img)
            lipstick_color[..., 0] = blue  # Blue channel
            lipstick_color[..., 1] = green  # Green channel
            lipstick_color[..., 2] = red  # Red channel
            if num_channels == 4:
                lipstick_color[..., 3] = 255  # 알파 채널을 255로 설정하여 색상이 완전히 불투명하도록 합니다.

            # Apply lipstick to the lip region
            img_with_lipstick = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
            lipstick_region = cv2.bitwise_and(lipstick_color, lipstick_color, mask=mask)
            img_with_lipstick += lipstick_region
            
    except Exception as e:
        # Log the error message if needed
        print(f"Error applying lipstick: {e}")

    return img_with_lipstick