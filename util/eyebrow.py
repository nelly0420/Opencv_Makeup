import cv2
import dlib
import numpy as np
import pandas as pd

def apply_eyebrow(image, prdCode):
    

    json_path = 'products.json'
    df = pd.read_json(json_path)

    match_row = df[df['prdCode'] == prdCode]
    if not match_row.empty:
        info = match_row[['color', 'option1', 'option2']]
        print(info)

        hex_color = info['color'].values[0].lstrip('#')
        new_color = tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))

    else:
        print("No Matching prdCode: {prdCode}")
        new_color = (0,0,0)
    

    # BGR로 변환
    bgr_color = (new_color[2], new_color[1], new_color[0])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # ---- 위는 공통영역 -------- # -> 희선파트

    for face in faces:
        shape = predictor(gray, face)

        left_eyebrow_points = []
        for i in range(17, 22):  # 입술의 외부의 랜드마크 포인트는 48-60
            x = shape.part(i).x
            y = shape.part(i).y
            left_eyebrow_points.append((x, y))

        right_eyebrow_points = []
        for i in range(22, 27):  # 입술의 외부의 랜드마크 포인트는 48-60
            x = shape.part(i).x
            y = shape.part(i).y
            right_eyebrow_points.append((x, y))

        right_eyebrow_points = np.array(right_eyebrow_points)
        left_eyebrow_points = np.array(left_eyebrow_points)
        
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(left_eyebrow_points)], color=bgr_color)
        cv2.fillPoly(overlay, [np.array(right_eyebrow_points)], color=bgr_color)

        # 투명도 조절
        image_with_eyebrow = cv2.addWeighted(image, 0.2, overlay, 0.8, 0)

        # 1) line 형식으로 눈썹에 맞게
        # cv2.polylines(image, [left_eyebrow_points[2:4]], isClosed=False, color=bgr_color, thickness=14)
        # cv2.polylines(image, [left_eyebrow_points[0:4]], isClosed=False, color=bgr_color, thickness=10)

        # # 오른쪽 눈썹 그리기
        # cv2.polylines(image, [right_eyebrow_points[2:4]], isClosed=False, color=bgr_color, thickness=4)
        # cv2.polylines(image, [right_eyebrow_points[0:2]], isClosed=False, color=bgr_color, thickness=2)
        # cv2.polylines(image, [right_eyebrow_points[4:]], isClosed=False, color=bgr_color, thickness=2)

        # 2) 점연결하는 방법말고 도형형태입히기도 생각해보기
        
    return image_with_eyebrow

## 구현을 위한 main code
if __name__ == "__main__":
    image_path = "image.jpg"
    img = cv2.imread(image_path)

    eyebrow_color = (186,144,101)
    img_with_eyebrow = apply_eyebrow(img, 'EB0001')

    cv2.imshow("Image with Eyebrow", img_with_eyebrow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## 구현을 위한 Video On main code
# if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)  # 카메라를 열기 (0은 기본 카메라)

    # if not cap.isOpened():
    #     print("카메라를 열 수 없습니다.")
    #     exit()

    # while True:
    #     ret, frame = cap.read()  # 카메라에서 프레임 읽기

    #     if not ret:
    #         print("카메라에서 프레임을 읽을 수 없습니다.")
    #         break

    #     # 얼굴에 눈썹 입히기
    #     frame_with_eyebrow = apply_eyebrow(frame, 'EB0001')

    #     # 결과 표시
    #     cv2.imshow('Frame with Eyebrow', frame_with_eyebrow)

    #     # 'q' 키를 누르면 종료
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # 작업 완료 후 해제
    # cap.release()
    # cv2.destroyAllWindows()