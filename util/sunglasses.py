import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from util.detect import get_landmarks, get_eyeCenter_points
from PIL import Image

## image load test
# img = Image.open("static\\images\\product\\디올 선글라스.png")

# print(img.size)
# img.show()

#------------------------------------------------------------------------#

def resize_sunglasses(sun_img, left_eye_center, right_eye_center):
    """
    선글라스 이미지를 눈 사이의 거리와 비율에 맞춰 크기 조정.
    
    :param sun_img: 선글라스 이미지
    :param left_eye_center: 왼쪽 눈 중심 좌표
    :param right_eye_center: 오른쪽 눈 중심 좌표
    :return: 크기 조정된 선글라스 이미지
    """
   # 두 눈 사이의 거리 계산
    eye_width = int(np.linalg.norm(np.array(right_eye_center) - np.array(left_eye_center)))

    # 선글라스 이미지의 비율 유지하며 크기 조정 (예를 들어 1.5배)
    scaling_factor = 4  # 비율을 조정하는 인자
    new_width = int(eye_width * scaling_factor)
    aspect_ratio = sun_img.shape[1] / sun_img.shape[0]
    new_height = int(new_width / aspect_ratio)

    resized_sunglasses = cv2.resize(sun_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_sunglasses

def overlay_image(background, overlay, x, y):
    """
    주어진 배경 이미지 위에 오버레이 이미지를 합성.
    
    :param background: 배경 이미지
    :param overlay: 오버레이 이미지
    :param x: 오버레이 이미지가 위치할 x 좌표
    :param y: 오버레이 이미지가 위치할 y 좌표
    :return: 합성된 이미지
    """
    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    # 오버레이 이미지가 배경 이미지를 넘지 않도록 클리핑
    if x + overlay_width > bg_width:
        overlay_width = bg_width - x
        overlay = overlay[:, :overlay_width]
    if y + overlay_height > bg_height:
        overlay_height = bg_height - y
        overlay = overlay[:overlay_height]

    if overlay.shape[2] == 4:
        # 오버레이 이미지가 알파 채널(투명도)을 가지고 있는 경우
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        for c in range(0, 3):
            background[y:y+overlay_height, x:x+overlay_width, c] = (
                alpha_overlay * overlay[:, :, c] +
                alpha_background * background[y:y+overlay_height, x:x+overlay_width, c]
            )
    else:
        # 오버레이 이미지에 알파 채널이 없는 경우
        background[y:y+overlay_height, x:x+overlay_width] = overlay

    return background

def load_image(prdCode):
    """static 폴더에서 prdCode에 해당하는 선글라스 이미지를 로드"""
    
    # JSON 파일 경로 설정 (현재 디렉토리 기준)
    json_path = 'products.json'

    try:
        # JSON 파일 읽기
        df = pd.read_json(json_path)
        match_row = df[df['prdCode'] == prdCode]
        
        if not match_row.empty:
            # 이미지 경로 추출 (static 이하의 상대 경로 그대로 사용)
            img_rel_path = match_row.iloc[0]['imgsrc']
            img_path = img_rel_path.lstrip('/')

            # 파일 존재 여부 확인
            if not os.path.exists(img_path):
                print(f"이미지 파일이 존재하지 않습니다: {img_path}")
                return None

            # PIL을 사용해 이미지 로드
            image = Image.open(img_path)
            #print("이미지 로드 성공!")

            # OpenCV에서 사용할 수 있도록 변환
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
            return image
        else:
            print(f"prdCode {prdCode}와 일치하는 행을 찾을 수 없습니다.")
            return None
    
    except Exception as e:
        print(f"JSON 파일을 읽는 동안 오류가 발생했습니다: {e}")
        return None

def apply_sunglasses(image, prdCode):
    
    sun_img = load_image(prdCode)
    if sun_img is None:
        print("선글라스 이미지를 로드할 수 없습니다.")
        return image

    landmarks = get_landmarks(image)
    if landmarks is None:
        print("얼굴 랜드마크를 감지할 수 없습니다.")
        return image  # 얼굴을 감지할 수 없는 경우 원본 이미지를 반환
    
    # # 눈의 중심 좌표 계산
    left_eye_center, right_eye_center = get_eyeCenter_points(landmarks)
    
    # # 눈 중앙에 빨간 둥근 점 표시
    # cv2.circle(image, tuple(left_eye_center), 2, (0, 0, 255), -1)
    # cv2.circle(image, tuple(right_eye_center), 2, (0, 0, 255), -1)
    
    
    # 선글라스 크기 조정
    resized_sunglasses = resize_sunglasses(sun_img, left_eye_center, right_eye_center)
    
    # 선글라스 이미지의 위치 설정
    eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)
    x_offset = eye_center[0] - resized_sunglasses.shape[1] // 2
    y_offset = eye_center[1] - resized_sunglasses.shape[0] // 2 + resized_sunglasses.shape[0] // 20
    
    # 선글라스 이미지를 얼굴 이미지에 합성
    result_image = overlay_image(image, resized_sunglasses, x_offset, y_offset)
    
    return result_image

# -------------> 선글라스 html 수정이 필요해서 test용으로 main code를 만듦.
# def main():
#     # 카메라 초기화
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("카메라를 열 수 없습니다.")
#         return
    
#     # 화면 크기 설정
#     window_name = 'Face with Sunglasses'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, 600, 450)
    
    
#     while True:
#         # 프레임 읽기
#         ret, frame = cap.read()
#         if not ret:
#             print("프레임을 읽을 수 없습니다.")
#             break
        

#         # 얼굴 랜드마크 및 눈 중앙 표시
#         processed_frame = apply_sunglasses(frame, 'SUN002')
        
#         # 결과 이미지 출력
#         cv2.imshow(window_name, processed_frame)
        
#         # 'q' 키를 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # 모든 창 닫기 및 카메라 해제
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()