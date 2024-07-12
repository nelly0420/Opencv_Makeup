import cv2
import dlib
import numpy as np
import pandas as pd



def apply_lipstick(image, prdCode):
    
    # < 수정파트>
    # 1. prdCode json에서 불러오기(ok)
    # 2. pandas를 사용해서 json을 읽어들이기 -> 데이터프레임 형태로(ok)
    # 3. 색상정보, option1, 옵션2(ok)
    # 4. json의 color를 rgb형태로 바꾼다.(ok)
    # 5. 공통인 부분 util.py에서 작성하기 -> 다른 부분부터 type 파일로 진행

    # JSON 불러오기
    json_path = 'products.json'
    df = pd.read_json(json_path)

    # print(df)

    # 색상정보, 옵션 1, 옵션2 가져오기
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # ---- 위는 공통영역 -------- # -> 희선파트

    for face in faces:
        shape = predictor(gray, face)

        # 입술 영역 좌표 추출
        lips_outer_points = []
        for i in range(48, 60):  # 입술의 외부의 랜드마크 포인트는 48-60
            x = shape.part(i).x
            y = shape.part(i).y
            lips_outer_points.append((x, y))
        
       # 입술 내부 영역 좌표 추출
        lips_inner_points = []
        for i in range(60, 68):  # 입술의 내부 랜드마크 포인트는 60-67
            x = shape.part(i).x
            y = shape.part(i).y
            lips_inner_points.append((x, y))
        
        lips_outer_points = np.array(lips_outer_points)
        lips_inner_points = np.array(lips_inner_points)

        # 입술 외부 마스크 생성
        mask_outer = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_outer, [lips_outer_points], 255)

        # 입술 내부 마스크 생성 (이빨 부분 제외)
        mask_inner = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_inner, [lips_inner_points], 255)

        # 이빨 부분 제외한 입술 영역 마스크 생성
        mask = cv2.subtract(mask_outer, mask_inner)

        # 입술 영역 색상 변경
        lips_region = cv2.bitwise_and(image, image, mask=mask)
        blurred_lips = cv2.GaussianBlur(lips_region, (9, 9), 0)

        # 새로운 립스틱 색상 적용
        new_lip_color = np.full_like(image, bgr_color, dtype=np.uint8)
        blended_lips = cv2.addWeighted(blurred_lips, 0.25, new_lip_color, 0.75, 0) # 매트타입

        # 블러 적용된 입술과 원래 이미지를 결합
        blurred_lips_result = cv2.add(blended_lips, blurred_lips) # 광택타입

        # 광택 효과 추가
        # lipgloss_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # cv2.fillPoly(lipgloss_mask, [lips_outer_points], 255)

        # lipgloss = np.zeros_like(image)
        # lipgloss[mask_outer != 0] = (255, 255, 255)  # 흰색으로 광택 부여
        # lips_with_gloss = cv2.addWeighted(blended_lips, 0.3, lipgloss, 0.7, 0)

        # 이미지에 새로운 입술 색상 적용
        image[np.where(mask == 255)] = blended_lips[np.where(mask == 255)]

    return image
    

## 구현을 위한 main code
if __name__ == "__main__":
    image_path = "tom.jpg"
    img = cv2.imread(image_path)
    # if img.shape != (1080, 1080, 3):
    #     img = cv2.resize(img, (1080, 1080))

    lipgloss_color = (207,70,76)  # 분홍색 계열
    img_with_lipstick = apply_lipstick(img, 'L00001')

    cv2.imshow("Image with Lipstick", img_with_lipstick)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
