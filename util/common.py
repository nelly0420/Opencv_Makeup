import dlib
import cv2

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None  # 얼굴이 감지되지 않았을 경우

    landmarks = predictor(gray, faces[0])  # 첫 번째 얼굴만 처리
    return landmarks

def get_eyebrows(landmarks):
    left_eyebrow_points = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22) ]
    right_eyebrow_points = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27) ]
    return left_eyebrow_points, right_eyebrow_points

def get_lips(landmarks):
    lip_points = [ (landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68) ]
    return lip_points