import cv2
import dlib
import numpy as np
import pandas as pd
# 색상 정보를 JSON 파일에서 가져오는 함수 만들기. -> util.py로 따로 저장해야함.
def get_color_from_json(prdCode):
    # JSON 불러오기
    json_path = 'products.json'
    df = pd.read_json(json_path)

    # 색상정보, 옵션 1, 옵션2 가져오기
    match_row = df[df['prdCode'] == prdCode]
    if not match_row.empty:
        info = match_row[['color', 'option1', 'option2']]
        print(info)

        hex_color = info['color'].values[0].lstrip('#')
        new_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        print(f"No Matching prdCode: {prdCode}")
        new_color = (0, 0, 0)
    
    return (new_color[2], new_color[1], new_color[0])  # BGR로 변환

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blush_alpha = 0.3  # 투명도
blush_offset = -20  # 볼 위치를 위로 이동시키는 오프셋 값 (픽셀 단위)
x_offset = 5  # 볼 위치를 왼쪽으로 이동시키는 오프셋 값 (픽셀 단위)

def get_cheek_landmarks(shape, y_offset, x_offset):
    # 얼굴의 중심을 계산-> 얼굴에 대칭적으로 볼터치를 입히기 위해서 얼굴의 중심을 계산하는거임.
    center_x = (shape.part(30).x + shape.part(8).x) // 2
    center_y = (shape.part(30).y + shape.part(8).y) // 2
    
    # 왼쪽과 오른쪽 볼의 인덱스
    left_cheek_idxs = [1, 2, 3, 4, 31, 49]
    right_cheek_idxs = [12, 13, 14, 15, 35, 53]
    
    # 좌우 구분하여 좌표 반환 및 y 좌표에 오프셋 적용
    left_cheek = np.array([[shape.part(i).x + x_offset, shape.part(i).y + y_offset] for i in left_cheek_idxs])
    right_cheek = np.array([[shape.part(i).x + x_offset, shape.part(i).y + y_offset] for i in right_cheek_idxs])
    
    return left_cheek, right_cheek

def apply_blush(image, prdCode):
    # 색상 정보를 JSON에서 가져오기
    blush_color = get_color_from_json(prdCode)

    # 이미지 BGRA로 변환
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    faces = detector(image, 1)
    if len(faces) == 0:
        print("No faces detected.")
        return image

    for k, d in enumerate(faces):
        # 얼굴 랜드마크 예측
        shape = predictor(image, d)
        
        left_cheek, right_cheek = get_cheek_landmarks(shape, blush_offset, x_offset)
        
        for cheek, name in zip([left_cheek, right_cheek], ["left_cheek", "right_cheek"]):
            # 볼 영역 마스크 생성 -> 볼터치 영역 적용
            mask = np.zeros(image.shape[:2], dtype=np.uint8)#-> 볼 영역만 채우게 하려고 볼터치영역 이외는 다 0으로 설정
            pts = cheek.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255) #-> 다각형을 채우는 함수

            # 블러쉬 컬러 채널 생성 및 적용-> 블러쉬 색 적용
            blush = np.zeros_like(image_bgra) #모든 값이 0으로 초기화, 아무런 색상이 적용되지 않음.
            blush[:, :, :3] = blush_color  # BGR 포맷
            blush[:, :, 3] = (mask * blush_alpha * 255).astype(np.uint8)  # 알파 채널 설정, 마스크 값이 255인 영역에만 블러쉬 색상이 적용

            # 가우시안 블러 적용
            blush[:, :, 3] = cv2.GaussianBlur(blush[:, :, 3], (25, 25), 0)  # 커널 크기 조정
            # 추가된 미디안 블러 적용
            blush[:, :, 3] = cv2.medianBlur(blush[:, :, 3], 7)  # 커널 크기 조정

            # 알파 채널 고려하여 최종 합성
            alpha_mask = blush[:, :, 3] / 255.0
            for c in range(0, 3):
                image[:, :, c] = image[:, :, c] * (1 - alpha_mask) + blush[:, :, c] * alpha_mask

    return image
