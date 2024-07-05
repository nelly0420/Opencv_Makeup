import cv2
import dlib
import numpy as np
from collections import OrderedDict
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 주위 랜드마크 인덱스 정의
EYE_IDXS = OrderedDict([
    ("left_eye", (36, 37, 38, 39, 40, 41)),
    ("right_eye", (42, 43, 44, 45, 46, 47))
])

# 아이라인 색상 및 두께
eyeline_color = (0, 0, 0)  # 검정색 (BGR 포맷)
eyeline_thickness = 2

# Flask 애플리케이션 정의
app = Flask(__name__, static_url_path="/static")
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_image(data):
    # Decode image from bytes
    nparr = np.frombuffer(data['image'], np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply eyeliner
    img_with_makeup = apply_eyeliner(img)

    # Encode image back to bytes and send back to client
    _, buffer = cv2.imencode('.jpg', img_with_makeup)
    emit('processed_image', {'image': buffer.tobytes()})

def apply_eyeliner(image):
    overlay = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)

    for k, d in enumerate(detections):
        shape = predictor(gray, d)

        for eye, (i1, i2) in EYE_IDXS.items():
            pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(i1, i2)])
            pts = pts.reshape((-1, 1, 2))

            # 눈 영역 추출 및 contours 찾기
            eye_region = gray[pts[0][0][1]:pts[-1][0][1], pts[:, 0, 0].min():pts[:, 0, 0].max()]
            _, threshold = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)

            # OpenCV 버전에 따른 contours 반환 처리
            contours = None
            if cv2.__version__.startswith('3'):
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elif cv2.__version__.startswith('4'):
                _, contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # contours를 원본 이미지에 그리기
            cv2.drawContours(overlay, contours, -1, eyeline_color, eyeline_thickness)

    return overlay
