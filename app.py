from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import dlib

# Application 정의
app = Flask(__name__, static_url_path='/static')

# 소켓 정의
socketio = SocketIO(app)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_image(data):
    # Decode image from bytes
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Detect faces
    faces = detector(img, 1)
    # Here you would process the image and apply color changes
    for face in faces:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    # Encode image back to bytes and send back to client
    _, buffer = cv2.imencode('.jpg', img)
    emit('processed_image', buffer.tobytes())

@app.route("/")
def main():
    return render_template("main.html")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
