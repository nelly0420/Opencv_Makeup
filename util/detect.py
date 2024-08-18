import dlib
import cv2
import numpy as np

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

def get_eyebrows(landmarks, scale_factor=1):
    """
    주어진 랜드마크에서 눈썹 포인트를 추출합니다.
    
    :param landmarks: dlib 랜드마크 객체
    :return: 왼쪽 및 오른쪽 눈썹 포인트 리스트
    """
    left_eyebrow_points = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22) ]
    right_eyebrow_points = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27) ]

    y_offset=2

    # 17번 포인트의 y 좌표를 y_offset 만큼 줄임
    left_eyebrow_points[0] = (left_eyebrow_points[0][0], left_eyebrow_points[0][1] - y_offset)

    # 27번 포인트의 y 좌표를 y_offset 만큼 줄임
    right_eyebrow_points[-1] = (right_eyebrow_points[-1][0], right_eyebrow_points[-1][1] - y_offset)


    # 왼쪽 눈썹의 중앙값 계산
    left_mean_x = np.mean([point[0] for point in left_eyebrow_points])
    # 오른쪽 눈썹의 중앙값 계산
    right_mean_x = np.mean([point[0] for point in right_eyebrow_points])

    # 각 눈썹 포인트의 x 좌표 조정
    left_eyebrow_points = [(int(left_mean_x + (x - left_mean_x) * scale_factor), y) for x, y in left_eyebrow_points]
    right_eyebrow_points = [(int(right_mean_x + (x - right_mean_x) * scale_factor), y) for x, y in right_eyebrow_points]

    right_eyebrow_points = np.array(right_eyebrow_points)
    left_eyebrow_points = np.array(left_eyebrow_points)
    
    return left_eyebrow_points, right_eyebrow_points

def get_lip_points(landmarks):
    """
    주어진 랜드마크에서 입술 포인트를 추출합니다.
    
    :param landmarks: dlib 랜드마크 객체
    :return: 입술 포인트 배열
    """
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # 윗입술 포인트
    indices_top = [64, 63, 62, 61, 60, 48, 49, 50, 51, 52, 53, 54]
    points_top = [points[i] for i in indices_top]

    # 아랫입술 포인트
    indices_bottom = [54, 55, 56, 57, 58, 59, 48, 60, 67, 66, 65, 64]
    points_bottom = [points[i] for i in indices_bottom]

    # 포인트 결합
    lip_points = np.array(points_top + points_bottom)
    return lip_points


def get_eyeCenter_points(landmarks):
    """
    얼굴 랜드마크 좌표를 입력받아 왼쪽 눈과 오른쪽 눈의 중심 좌표를 반환합니다.

    """
    left_eye_idxs = [36, 37, 38, 39, 40, 41]
    right_eye_idxs = [42, 43, 44, 45, 46, 47]
    
    left_eye_x = int(np.mean([landmarks.part(i).x for i in left_eye_idxs]))
    left_eye_y = int(np.mean([landmarks.part(i).y for i in left_eye_idxs]))
    left_eye_center = (left_eye_x, left_eye_y)
    
    right_eye_x = int(np.mean([landmarks.part(i).x for i in right_eye_idxs]))
    right_eye_y = int(np.mean([landmarks.part(i).y for i in right_eye_idxs]))
    right_eye_center = (right_eye_x, right_eye_y)
    
    return left_eye_center, right_eye_center


def get_cheek_centers(shape, y_offset, x_offset):
    """
    주어진 랜드마크에서 왼쪽 및 오른쪽 뺨의 중심을 계산합니다.
    
    :param shape: dlib 랜드마크 객체
    :param y_offset: y축 오프셋
    :param x_offset: x축 오프셋
    :return: 왼쪽 및 오른쪽 뺨 중심 좌표
    """
    left_cheek_idxs = [1, 2, 3, 4, 31, 49]
    right_cheek_idxs = [12, 13, 14, 15, 35, 53]
    
    left_cheek_x = np.mean([shape.part(i).x for i in left_cheek_idxs]) + x_offset
    left_cheek_y = np.mean([shape.part(i).y for i in left_cheek_idxs]) + y_offset
    left_cheek_center = np.array([left_cheek_x, left_cheek_y])
    
    right_cheek_x = np.mean([shape.part(i).x for i in right_cheek_idxs]) + x_offset
    right_cheek_y = np.mean([shape.part(i).y for i in right_cheek_idxs]) + y_offset
    right_cheek_center = np.array([right_cheek_x, right_cheek_y])
    
    return left_cheek_center, right_cheek_center

def get_eye_points(landmarks, eye_indices):
    """
    주어진 랜드마크에서 눈의 포인트를 추출합니다.
    
    :param landmarks: dlib 랜드마크 객체
    :param eye_indices: 눈의 포인트 인덱스
    :return: 눈 포인트 리스트
    """
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]

def create_shifted_points(points, y_shift):
    """
    주어진 포인트를 y축으로 이동시킵니다.
    
    :param points: 원본 포인트 리스트
    :param y_shift: 이동 거리
    :return: 이동된 포인트 리스트
    """
    return [(x, y - y_shift) for (x, y) in points]

def add_intermediate_points(points, num_points=1):
    """
    두 포인트 사이에 중간 포인트를 추가합니다.
    
    :param points: 원본 포인트 리스트
    :param num_points: 추가할 중간 포인트 수
    :return: 중간 포인트가 추가된 포인트 리스트
    """
    detailed_points = []
    for i in range(len(points) - 1):
        detailed_points.append(points[i])
        for j in range(1, num_points + 1):
            mid_point = (
                points[i][0] + (points[i + 1][0] - points[i][0]) * j // (num_points + 1),
                points[i][1] + (points[i + 1][1] - points[i][1]) * j // (num_points + 1),
            )
            detailed_points.append(mid_point)
    detailed_points.append(points[-1])
    return np.array(detailed_points, dtype=np.int32)

def expand_points(points, factor=1.2):
    """
    주어진 포인트를 중심으로 확대합니다.
    
    :param points: 원본 포인트 리스트
    :param factor: 확대 배율
    :return: 확대된 포인트 리스트
    """
    center_x = np.mean([x for (x, y) in points])
    center_y = np.mean([y for (x, y) in points])
    expanded_points = [(int(center_x + (x - center_x) * factor), int(center_y + (y - center_y) * factor)) for (x, y) in points]
    return expanded_points
