import cv2
import dlib
import numpy as np
from collections import OrderedDict

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 주위 랜드마크 인덱스 정의
EYE_IDXS = OrderedDict([
    ("left_eye", list(range(36, 42))),  # 왼쪽 눈의 랜드마크 인덱스
    ("right_eye", list(range(42, 48)))  # 오른쪽 눈의 랜드마크 인덱스
])

# 아이라인 색상 및 두께 (브라운색)
eyeline_color = (51, 25, 0)  # 브라운색 (BGR 포맷)
eyeline_thickness = 1  # 선의 두께
eyeline_alpha = 0.3  # 투명도

def smooth_polyline(image, points, thickness):
    num_points = len(points)
    if num_points < 2:
        return

    for i in range(num_points - 1):
        p1 = points[i]
        p2 = points[i + 1]
        cv2.line(image, p1, p2, eyeline_color, thickness)

def apply_eyeliner(image):
    image_copy = image.copy()  # 이미지의 복사본 생성
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        print("No faces detected.")
        return image_copy

    for k, d in enumerate(faces):
        shape = predictor(gray, d)

        for eye, indices in EYE_IDXS.items():
            points = [(shape.part(i).x, shape.part(i).y) for i in indices]

            # 눈 주위 점들 중 최소 y 좌표 찾기
            eye_top = np.min([point[1] for point in points])

            # 윗부분 점들만 선택하여 아이라인을 그림
            upper_points = [(shape.part(i).x, shape.part(i).y) for i in indices if shape.part(i).y <= eye_top + 3]

            if len(upper_points) > 1:
                # 부드러운 곡선을 그리기 위해 점들을 연결
                smooth_polyline(image_copy, upper_points, eyeline_thickness)

                # 그라데이션 효과 적용
                gradient_length = len(upper_points) - 1
                for i in range(gradient_length):
                    p1 = upper_points[i]
                    p2 = upper_points[i + 1]

                    # 선의 끝 부분에서 점점 투명해지도록 계산
                    for j in range(eyeline_thickness):
                        alpha = int((j / eyeline_thickness) * eyeline_alpha * 255)  # 그라데이션 투명도 계산
                        overlay = image_copy.copy()
                        cv2.line(overlay, p1, p2, (*eyeline_color, alpha), 1)
                        image_copy = cv2.addWeighted(overlay, 1 - eyeline_alpha, image_copy, eyeline_alpha, 0)

    return image_copy



