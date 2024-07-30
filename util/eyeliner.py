import cv2
import dlib
import numpy as np
from collections import OrderedDict
from util.utils import get_color_from_json  # util.py에서 get_color_from_json 함수를 import

# dlib 초기화: 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 주위 랜드마크 인덱스 정의
EYE_IDXS = OrderedDict([
    ("left_eye", list(range(36, 42))),  # 왼쪽 눈의 랜드마크 인덱스
    ("right_eye", list(range(42, 48)))  # 오른쪽 눈의 랜드마크 인덱스
])

# 아이라인 색상 및 두께
eyeline_thickness = 2  # 아이라인의 두께 설정 (픽셀 단위)
eyeline_alpha = 0.2  # 아이라인의 투명도 (0.0 ~ 1.0)
eyeline_offset_y = -5  # 아이라인의 y축 위치 조정

def bezier_curve(points, n=100):
    """
    De Casteljau's algorithm을 사용하여 베지어 곡선을 생성합니다.
    
    Parameters:
        points (numpy.array): 곡선을 정의하는 제어 점들
        n (int): 곡선을 구성하는 점의 수
    
    Returns:
        numpy.array: 베지어 곡선의 점들
    """
    t_values = np.linspace(0, 1, n)  # 0에서 1 사이의 n개의 등간격 t 값 생성
    curve = np.zeros((n, 2))  # 곡선의 점들을 저장할 배열

    for i, t in enumerate(t_values):
        temp_points = np.copy(points)
        while len(temp_points) > 1:
            temp_points = (1 - t) * temp_points[:-1] + t * temp_points[1:]
        curve[i] = temp_points[0]

    return curve.astype(int)

def apply_eyeliner(image, prdCode):
    """
    이미지에 아이라인을 적용합니다.
    
    Parameters:
        image (numpy.array): 입력 이미지
        prdCode (str): 색상 정보를 얻기 위한 코드
    
    Returns:
        numpy.array: 아이라인이 적용된 이미지
    """
    # 색상 정보를 JSON에서 가져오기
    eyeline_color, option = get_color_from_json(prdCode)

    image_copy = image.copy()  # 원본 이미지의 복사본 생성
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)  # 이미지의 회색조 변환
    faces = detector(gray, 0)  # 얼굴 감지

    if len(faces) == 0:
        print("No faces detected.")
        return image_copy

    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(gray, d)

        for eye, indices in EYE_IDXS.items():
            # 눈의 랜드마크 좌표 추출 및 y축 위치 조정
            points = [(shape.part(i).x, shape.part(i).y + eyeline_offset_y) for i in indices]

            # 눈의 상부 랜드마크 추출
            upper_points = points[:4]  # 상부 4개의 랜드마크 점

            if len(upper_points) > 1:
                # 베지어 곡선을 생성하기 위해 상부 점들을 연결
                curve = bezier_curve(np.array(upper_points))

                # 곡선을 이미지에 그리기
                for i in range(len(curve) - 1):
                    cv2.line(image_copy, tuple(curve[i]), tuple(curve[i + 1]), eyeline_color, eyeline_thickness)

                # 그라데이션 효과 적용
                gradient_length = len(curve) - 1
                for i in range(gradient_length):
                    p1 = curve[i]
                    p2 = curve[i + 1]

                    # 선의 끝 부분에서 점점 투명해지도록 계산
                    overlay = image_copy.copy()
                    cv2.line(overlay, p1, p2, eyeline_color, eyeline_thickness)
                    image_copy = cv2.addWeighted(overlay, eyeline_alpha, image_copy, 1 - eyeline_alpha, 0)

    # 가우시안 블러 적용 (최종 이미지를 부드럽게 만듭니다)
    result_image = cv2.GaussianBlur(image_copy, (5, 5), 0)

    return result_image
