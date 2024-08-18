import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    pupils = []
    for face in faces:
        shape = predictor(gray, face)
        left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 40)]
        right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 46)]

        # 동공 위치 조정 영역
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

        left_eye_radius = int(np.linalg.norm(np.array(left_eye_points[1]) - np.array(left_eye_points[3])) / 2)
        right_eye_radius = int(np.linalg.norm(np.array(right_eye_points[1]) - np.array(right_eye_points[3])) / 2)

        # 동공을 찾기 위한 이진화
        def find_pupils(eye_center, eye_radius, eye_points, y_offset, radius_factor):
            mask = np.zeros_like(gray)
            cv2.circle(mask, tuple(eye_center), eye_radius, 255, -1)

            eye_roi = cv2.bitwise_and(gray, gray, mask=mask)

            _, thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)

            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_pupils = []

            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    y_adjusted = int(y + y_offset)  # Y축 조정
                    r_adjusted = int(radius * radius_factor)  # 반지름 조정
                    detected_pupils.append(((int(x + eye_center[0]), int(y_adjusted + eye_center[1])), r_adjusted))

            return detected_pupils

        # 동공 위치와 반지름 조정
        pupils.extend(find_pupils(left_eye_center, left_eye_radius, left_eye_points, y_offset=40, radius_factor=0.03))
        pupils.extend(find_pupils(right_eye_center, right_eye_radius, right_eye_points, y_offset=40, radius_factor=0.03))

    return pupils

def draw_pupil_borders(image, pupils, color):
    for (center, radius) in pupils:
        # 동공 테두리 그리기
        cv2.circle(image, center, radius, color, thickness=1)  # thickness=3으로 테두리 두께 조정
    return image

def apply_lens(image_path, prdCode):
    image = cv2.imread(image_path)
    pupils = detect_pupil(image)
    
    if pupils:
        # 색상 정보 가져오기
        bgr_color_str, _ = get_color_from_json(prdCode)  # JSON에서 색상 정보 가져오기
        if bgr_color_str:
            # BGR 색상값을 튜플로 변환
            bgr_color = tuple(map(int, bgr_color_str.split(',')))
            image_with_borders = draw_pupil_borders(image, pupils, bgr_color)
            
            # 결과 이미지 반환
            return image_with_borders
        else:
            print("Invalid color code.")
            return None
    else:
        print("No pupils detected.")
        return None
