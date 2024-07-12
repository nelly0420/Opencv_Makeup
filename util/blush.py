import cv2
import dlib
import numpy as np
from collections import OrderedDict

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 진한 핑크색 및 투명도
blush_color = (180, 105, 255)  # 진한 핑크 (BGR 포맷)
blush_alpha = 0.3  # 투명도
blush_offset = -10  # 볼 위치를 위로 이동시키는 오프셋 값 (픽셀 단위)

def get_cheek_landmarks(shape, offset):
    # 얼굴의 중심을 계산
    center_x = (shape.part(30).x + shape.part(8).x) // 2
    center_y = (shape.part(30).y + shape.part(8).y) // 2
    
    # 왼쪽과 오른쪽 볼의 인덱스
    left_cheek_idxs = [1, 2, 3, 4, 31, 49]
    right_cheek_idxs = [12, 13, 14, 15, 35, 53]
    
    # 좌우 구분하여 좌표 반환 및 y 좌표에 오프셋 적용
    left_cheek = np.array([[shape.part(i).x, shape.part(i).y + offset] for i in left_cheek_idxs])
    right_cheek = np.array([[shape.part(i).x, shape.part(i).y + offset] for i in right_cheek_idxs])
    
    return left_cheek, right_cheek

def apply_blush(image):
    # 이미지 BGRA로 변환
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    faces = detector(image, 1)
    if len(faces) == 0:
        print("No faces detected.")
        return image

    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(image, d)
        
        left_cheek, right_cheek = get_cheek_landmarks(shape, blush_offset)
        
        for cheek, name in zip([left_cheek, right_cheek], ["left_cheek", "right_cheek"]):
            # 볼 영역 마스크 생성
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            pts = cheek.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

            # 블러쉬 컬러 채널 생성 및 적용
            blush = np.zeros_like(image_bgra)
            blush[:, :, :3] = blush_color  # BGR 포맷
            blush[:, :, 3] = (mask * blush_alpha * 255).astype(np.uint8)  # 알파 채널 설정

            # 가우시안 블러 적용
            blush[:, :, 3] = cv2.GaussianBlur(blush[:, :, 3], (25, 25), 0)  # 커널 크기 조정
            # 추가된 미디안 블러 적용
            blush[:, :, 3] = cv2.medianBlur(blush[:, :, 3], 7)  # 커널 크기 조정

            # 알파 채널 고려하여 최종 합성
            alpha_mask = blush[:, :, 3] / 255.0
            for c in range(0, 3):
                image[:, :, c] = image[:, :, c] * (1 - alpha_mask) + blush[:, :, c] * alpha_mask

    return image

# import cv2
# import dlib
# import numpy as np
# from collections import OrderedDict

# # dlib 초기화
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # 볼 부분 랜드마크 인덱스 정의
# CHEEK_IDXS = OrderedDict([
#     ("left_cheek", [1, 2, 3, 4, 49, 31]),
#     ("right_cheek", [ 12, 13, 14, 15, 35, 53])
# ]) #하드코딩하면 안됨 바꿔야함. 점을 받은 상태에서 좌우를 조정하는 형태로 바꿔주기.

# # 블러쉬 색상 및 투명도
# blush_color = (193, 153, 255)  # 연한 분홍색 계열 (BGR 포맷)
# blush_alpha = 0.3  # 투명도

# def apply_blush(image):
#     # 이미지 BGRA로 변환
#     image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
#     faces = detector(image, 1)
#     if len(faces) == 0:
#         print("No faces detected.")
#         return image

#     for k, d in enumerate(faces):
#         # 얼굴 랜드마크 예측
#         shape = predictor(image, d)
        
#         for (_, name) in enumerate(CHEEK_IDXS.keys()):
#             pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32)
#             for i, j in enumerate(CHEEK_IDXS[name]):
#                 pts[i] = [shape.part(j).x, shape.part(j).y]

#             # 볼 영역 마스크 생성
#             mask = np.zeros(image.shape[:2], dtype=np.uint8)
#             pts = pts.reshape((-1, 1, 2))
#             cv2.fillPoly(mask, [pts], 255)

#             # 블러쉬 컬러 채널 생성 및 적용 -> 코드 다시 알아오기.
#             blush = np.zeros_like(image_bgra) 
#             blush[:, :, :3] = blush_color  # BGR 포맷
#             blush[:, :, 3] = (mask * blush_alpha * 255).astype(np.uint8)  # 알파 채널 설정

#             # 가우시안 블러 적용
#             blush[:, :, 3] = cv2.GaussianBlur(blush[:, :, 3], (25, 25), 0)  # 커널 크기 조정
#             # 추가된 미디안 블러 적용
#             blush[:, :, 3] = cv2.medianBlur(blush[:, :, 3], 7)  # 커널 크기 조정

#             # 알파 채널 고려하여 최종 합성
#             alpha_mask = blush[:, :, 3] / 255.0
#             for c in range(0, 3):
#                 image[:, :, c] = image[:, :, c] * (1 - alpha_mask) + blush[:, :, c] * alpha_mask

#     return image