import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json


# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1.2):
    dst = src.copy()
    dst = dst / 255.
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

def apply_lipstick(image, prdCode):
    lip_color, option = get_color_from_json(prdCode)
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected.")
        return image
    
    for face in faces:
        shape = predictor(gray, face)
        
        points = []
        for i in range(shape.num_parts):
            points.append((shape.part(i).x, shape.part(i).y))

        # 윗입술 포인트
        indices_top = [64, 63, 62, 61, 60, 48, 49, 50, 51, 52, 53, 54]
        points_top = [points[i] for i in indices_top]

        # 아랫입술 포인트
        indices_bottom = [54, 55, 56, 57, 58, 59, 48, 60, 67, 66, 65, 64]
        points_bottom = [points[i] for i in indices_bottom]

        # 포인트 결합
        lip_points = np.array(points_top + points_bottom)

        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, [lip_points], lip_color)

        # mask = cv2.GaussianBlur(mask, (7, 7), 5)

        # image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)

        # lips_only = np.zeros_like(image)
        # #lips_only = cv2.fillPoly(lips_only, [lip_points], (180, 229, 255))
        
        # if option == "Glossy":
        #     corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)
        #     corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 3)
        # elif option == "Matte":
        #     #corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)
        #     corrected_lips = cv2.GaussianBlur(lips_only, (5, 5), 0)
            
        # final_image = np.where(lips_only == np.array([255, 255, 255]), corrected_lips, image_with_lipstick)
        
        # (오리지날)
        # if option == "Glossy":
        #     mask = cv2.GaussianBlur(mask, (7, 7), 5)
        #     image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)
        #     corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)
        #     corrected_lips = cv2.GaussianBlur(corrected_lips, (5, 5), 3)
        
        # (수정된소스 : 립스틱 윤각을 더 강조하고 블러효과를 좀 감소시킴.)
        # if option == "Glossy":
        #     # 마스크에 경계 강조 적용 (경계 검출)
        #     edges = cv2.Canny(mask, 100, 200)  # Canny edge detector로 경계 강조
        #     edges = cv2.dilate(edges, None)  # 경계 확대

        #     # 경계 강조된 마스크를 원래 마스크와 결합
        #     mask = cv2.bitwise_or(mask, edges)

        #     # 이미지와 마스크 결합
        #     image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.5, 0.0)  # 가중치를 높임0.4 -> 0.5
        #     corrected_lips = gamma_correction(image_with_lipstick, gamma=1.2)

        # elif option == "Matte":
        #     #print("mask dtype:", mask.dtype)
        #     image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.4, 0.0)
        #     corrected_lips = gamma_correction(image_with_lipstick, gamma=1.5)
        #     corrected_lips = cv2.addWeighted(corrected_lips, 1.5, corrected_lips, -0.5, 0)
        
        # 윤곽선 추출
        contour_mask = np.zeros_like(image)
        contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_mask, contours, -1, lip_color, 1) 
        
        # 윤곽선에 블러효과 적용
        lip_liner = cv2.GaussianBlur(contour_mask, (15, 15), 0)
        

        if option == "Glossy":
            image_with_lipstick = cv2.addWeighted(image, 1.0, mask, 0.1, 0.0)
            corrected_lips = gamma_correction(image_with_lipstick, gamma= 1.2)
            corrected_lips = cv2.GaussianBlur(corrected_lips, (15, 15), 4) 
            lip_liner = gamma_correction(lip_liner, 0.8, 1.5)
            corrected_lips = cv2.addWeighted(corrected_lips, 0.8, lip_liner, 0.2, 0.0)

            # 반짝이는 효과를 위해 고정된 위치에 작은 하이라이트 추가
            # highlight_color = (255, 223, 186)  # 연한 골드색
            
            
            corrected_lips = cv2.addWeighted(corrected_lips, 1.2, corrected_lips, -0.2, 0)  


        ## 매트 윤곽선만 블러처리를 위한 코드
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
        #final_image = np.where(blurred_contour_mask == np.array([255, 255, 255]), blurred_contour_mask, corrected_lips)

        return final_image
