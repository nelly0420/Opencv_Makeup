import cv2
import dlib
import numpy as np
from util.utils import get_color_from_json

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
eyeline_alpha = 0.8  # 아이라인의 투명도 (0.0 ~ 1.0)
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

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert HEX color code to RGB color format."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Convert HEX to RGB
    return rgb  # Return RGB

def apply_eyeliner(image: np.ndarray, prdCode: str, color: str) -> np.ndarray:
    """
    이미지에 아이라인을 적용합니다.
    
    Parameters:
        image (numpy.array): 입력 이미지 (BGR 형식)
        prdCode (str): 색상 정보를 얻기 위한 코드
        color (str): 아이라이너 색상 (HEX 코드)
    
    Returns:
        numpy.array: 아이라인이 적용된 이미지 (BGR 형식)
    """
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 색상 정보를 JSON에서 가져오기
    eyeline_color_hex, _, _ = get_color_from_json(prdCode)
    eyeline_color = hex_to_rgb(color)  # HEX 색상 코드를 RGB로 변환

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # 이미지의 회색조 변환
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

                # Create a mask for the eyeliner
                mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
                cv2.polylines(mask, [curve], isClosed=False, color=255, thickness=eyeline_thickness)
                
                # Apply Gaussian blur to the mask to create a smooth eyeliner effect
                alpha_mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Smooth the mask to create a gradient effect
                alpha_mask = alpha_mask / 255.0  # Normalize to [0, 1] range

                # Convert image to float32 for blending
                image_rgb = image_rgb.astype(np.float32) / 255.0

                # Blend the color into the image using the alpha mask
                for c in range(3):
                    image_rgb[:, :, c] = image_rgb[:, :, c] * (1 - alpha_mask) + eyeline_color[c] / 255.0 * alpha_mask

                # Convert back to uint8
                image_rgb = (image_rgb * 255).astype(np.uint8)

    # Convert the final result back to BGR
    result_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return result_image
