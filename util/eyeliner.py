import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json  # util.py에서 get_color_from_json 함수를 import

# dlib 초기화: 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 주위 랜드마크 인덱스 정의
EYE_IDXS = {
    "left_eye": list(range(36, 42)),  # 왼쪽 눈의 랜드마크 인덱스
    "right_eye": list(range(42, 48))  # 오른쪽 눈의 랜드마크 인덱스
}

# 아이라인 색상 및 두께
eyeline_thickness = 1  # 아이라인의 두께 설정 (픽셀 단위)
eyeline_alpha = 0.2  # 아이라인의 투명도 (0.0 ~ 1.0)
eyeline_offset_y = -3  # 아이라인의 y축 위치 조정

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
    eyeline_color, _ = get_color_from_json(prdCode)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지의 회색조 변환
    faces = detector(gray, 0)  # 얼굴 감지

    if len(faces) == 0:
        print("No faces detected.")
        return image

    for d in faces:
        shape = predictor(gray, d)
        
        for indices in EYE_IDXS.values():
            points = np.array([(shape.part(i).x, shape.part(i).y + eyeline_offset_y) for i in indices])

            # 눈의 상부 랜드마크 추출
            upper_points = points[:4]  # 상부 4개의 랜드마크 점

            if len(upper_points) > 1:
                curve = bezier_curve(upper_points)

                # 곡선을 이미지에 그리기
                cv2.polylines(image, [curve], isClosed=False, color=eyeline_color, thickness=eyeline_thickness)

                # 그라데이션 효과 적용
                for i in range(len(curve) - 1):
                    p1, p2 = curve[i], curve[i + 1]
                    cv2.line(image, tuple(p1), tuple(p2), eyeline_color, eyeline_thickness)
                    alpha = eyeline_alpha * (i / len(curve))
                    overlay = image.copy()
                    cv2.line(overlay, tuple(p1), tuple(p2), eyeline_color, eyeline_thickness)
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 가우시안 블러 적용 (최종 이미지를 부드럽게 만듭니다)
    result_image = cv2.GaussianBlur(image, (5, 5), 0)

    return result_image
