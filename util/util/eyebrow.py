import cv2
import dlib
import numpy as np

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 입술 랜드마크 인덱스 정의 (이빨 부분 제외)
LIPS_IDXS = list(range(48, 60))

# 입술 영역 확장을 위한 오프셋
OFFSET_X = 2  # 좌우 확장 픽셀 단위
OFFSET_Y = 2  # 상하 확장 픽셀 단위

# 립글로스 색상 및 투명도
lipgloss_color = (255, 51, 153)  # 분홍색 계열 (BGR 포맷)
lipgloss_alpha = 0.3  # 투명도

def apply_eyebrow(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        print("No faces detected.")
    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(image, d)
        
        # 입술 영역 추출 및 확장
        pts = np.zeros((len(LIPS_IDXS), 2), np.int32)
        for i, j in enumerate(LIPS_IDXS):
            x, y = shape.part(j).x, shape.part(j).y
            if j in range(48, 55):  # 위쪽 입술 포인트 확장
                y -= OFFSET_Y
            elif j in range(55, 61):  # 아래쪽 입술 포인트 확장
                y += OFFSET_Y
            if j == 48:  # 왼쪽 끝점 확장
                x -= OFFSET_X
            if j == 54:  # 오른쪽 끝점 확장
                x += OFFSET_X
            pts[i] = [x, y]

        # 입술 영역 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # 입술 영역 추출
        lips = cv2.bitwise_and(image, image, mask=mask)

        # 입술 영역만 포함하는 이미지로 크롭
        x, y, w, h = cv2.boundingRect(pts)
        cropped_lips = lips[y:y+h, x:x+w]

        # 입술 영역에 색 입히기
        lips_color = np.zeros_like(image)
        lips_color[:] = lipgloss_color[::-1]  # BGR -> RGB로 변환
        lips_colored = cv2.bitwise_and(lips_color, lips_color, mask=mask)

        # 원본 이미지에서 입술 영역만 추출
        lips_original = cv2.bitwise_and(image, image, mask=mask)

        # 광택 효과 추가
        lipgloss_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(lipgloss_mask, [pts], 255)
        lipgloss = np.zeros_like(image)
        lipgloss[lipgloss_mask != 0] = (255, 255, 255)  # 흰색으로 광택 부여
        lips_with_gloss = cv2.addWeighted(lipgloss, lipgloss_alpha, lips_colored, 1 - lipgloss_alpha, 0)

        # 혼합 이미지 생성
        blended_lips = cv2.addWeighted(lips_with_gloss, 0.6, lips_original, 0.4, 0)

        # 원본 이미지에 입술 영역만 덮어쓰기
        mask_inv = cv2.bitwise_not(mask)
        image_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        image_with_gloss = cv2.add(image_bg, blended_lips)
        
        return image_with_gloss

    return image  # 얼굴을 찾지 못한 경우 원본 이미지 반환
