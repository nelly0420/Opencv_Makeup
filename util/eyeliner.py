import cv2
import dlib
import numpy as np
from collections import OrderedDict
from util.utils import get_color_from_json  # util.py에서 get_color_from_json 함수를 import

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 주위 랜드마크 인덱스 정의
EYE_IDXS = OrderedDict([
    ("left_eye", list(range(36, 42))),  # 왼쪽 눈의 랜드마크 인덱스
    ("right_eye", list(range(42, 48)))  # 오른쪽 눈의 랜드마크 인덱스
])

# 아이라인 색상 및 두께
eyeline_thickness = 2  # 선의 두께를 조금 굵게 설정
eyeline_alpha = 0.3  # 투명도
eyeline_offset_y = -4.5  # y 좌표를 위로 이동시키는 오프셋 값

def bezier_curve(points, n=100):
    # De Casteljau's algorithm for Bezier curves
    t_values = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))

    for i, t in enumerate(t_values):
        temp_points = np.copy(points)
        while len(temp_points) > 1:
            temp_points = (1 - t) * temp_points[:-1] + t * temp_points[1:]
        curve[i] = temp_points[0]

    return curve.astype(int)

def apply_eyeliner(image, prdCode):
    # 색상 정보를 JSON에서 가져오기
    eyeline_color = get_color_from_json(prdCode)

    image_copy = image.copy()  # 이미지의 복사본 생성
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        print("No faces detected.")
        return image_copy

    for k, d in enumerate(faces):
        shape = predictor(gray, d)

        for eye, indices in EYE_IDXS.items():
            points = [(shape.part(i).x, shape.part(i).y + eyeline_offset_y) for i in indices]

            # 눈의 상부 랜드마크 추출
            upper_points = points[:4]  # 왼쪽, 중앙 왼쪽, 중앙 오른쪽, 오른쪽 상부 점

            if len(upper_points) > 1: 
                # 베지어 곡선을 생성하기 위해 점들을 연결
                curve = bezier_curve(upper_points)

                # 곡선을 이미지에 그리기
                for i in range(len(curve) - 1):
                    cv2.line(image_copy, tuple(curve[i]), tuple(curve[i + 1]), eyeline_color, eyeline_thickness)

                # 그라데이션 효과 적용
                gradient_length = len(curve) - 1
                for i in range(gradient_length):
                    p1 = curve[i]
                    p2 = curve[i + 1]

                    # 선의 끝 부분에서 점점 투명해지도록 계산
                    for j in range(eyeline_thickness):
                        alpha = int((j / eyeline_thickness) * eyeline_alpha * 255)  # 그라데이션 투명도 계산
                        overlay = image_copy.copy()
                        cv2.line(overlay, p1, p2, (*eyeline_color, alpha), 1)
                        image_copy = cv2.addWeighted(overlay, 1 - eyeline_alpha, image_copy, eyeline_alpha, 0)

    return image_copy