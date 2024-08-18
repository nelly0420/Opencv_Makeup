import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from util.detect import get_landmarks, get_eyeCenter_points

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    """
    주어진 이미지에서 얼굴의 랜드마크를 감지합니다.
    
    :param image: 입력 이미지 (BGR 포맷)
    :return: 얼굴 랜드마크의 dlib 객체, 얼굴이 감지되지 않았을 경우 None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None  # 얼굴이 감지되지 않았을 경우
    landmarks = predictor(gray, faces[0])  # 첫 번째 얼굴만 처리
    return landmarks


def get_eyeCenter_points(landmarks):
    """
    얼굴 랜드마크 좌표를 입력받아 왼쪽 눈과 오른쪽 눈의 중심 좌표를 반환합니다.

    """
    # 왼쪽 눈의 중심 좌표 계산
    left_eye = np.mean([np.array([landmarks.part(37).x, landmarks.part(37).y]),
                        np.array([landmarks.part(38).x, landmarks.part(38).y]),
                        np.array([landmarks.part(40).x, landmarks.part(40).y]),
                        np.array([landmarks.part(41).x, landmarks.part(41).y])], axis=0).astype(int)
    
    # 오른쪽 눈의 중심 좌표 계산
    right_eye = np.mean([np.array([landmarks.part(43).x, landmarks.part(43).y]),
                         np.array([landmarks.part(44).x, landmarks.part(44).y]),
                         np.array([landmarks.part(46).x, landmarks.part(46).y]),
                         np.array([landmarks.part(47).x, landmarks.part(47).y])], axis=0).astype(int)
    
    # 리스트 형식으로 반환
    return list(left_eye), list(right_eye)

def load_image(prdCode):
    """products.json파일에 있는 선글라스 사진 load"""

    json_path = 'products.json'  # JSON 파일 경로
    df = pd.read_json(json_path)  # JSON 파일 읽기

    match_row = df[df['prdCode'] == prdCode]  # prdCode와 일치하는 행 찾기
    if not match_row.empty:
        img_path = match_row.iloc[0]['imgsrc']  # 이미지 경로 추출
        glasses = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if glasses is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
        return glasses
    else:
        print(f"prdCode {prdCode}와 일치하는 행을 찾을 수 없습니다.")
        return None

def apply_sunglasses(image, prdCode):
    
    sun_img = load_image(prdCode)
    left_eyeglass = np.asarray([200, 115])
    right_eyeglass = np.asarray([615, 115])


    ## Draw center of eyeglass per eye
    clone_glasses = sun_img.copy()
    cv2.circle(clone_glasses, tuple(left_eyeglass), 2, (0, 0, 255), -1)
    cv2.circle(clone_glasses, tuple(right_eyeglass), 2, (0, 0, 255), -1)
    clone_glasses = cv2.cvtColor(clone_glasses,cv2.COLOR_BGR2RGB)
    plt.imshow(clone_glasses)
    

     # 얼굴 랜드마크를 감지
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        print("얼굴이 감지되지 않았습니다.")
        return image
    
    # 눈의 중심 좌표 계산
    left_eye_center, right_eye_center = get_eyeCenter_points(landmarks)
    
    # 눈 중앙에 빨간 둥근 점 표시
    cv2.circle(image, tuple(left_eye_center), 1, (0, 0, 255), -1)
    cv2.circle(image, tuple(right_eye_center), 1, (0, 0, 255), -1)
    
    rslt_img = image

    return rslt_img

def main():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    # 화면 크기 설정
    window_name = 'Face with Sunglasses'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 450)
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 얼굴 랜드마크 및 눈 중앙 표시
        processed_frame = apply_sunglasses(frame, 'SUN002')
        
        # 결과 이미지 출력
        cv2.imshow(window_name, processed_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 모든 창 닫기 및 카메라 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()