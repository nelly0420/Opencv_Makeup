from flask import Flask, render_template, redirect, url_for
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
@app.route("/main")
def main():
    return render_template("main.html")
    

@app.route("/products_skin")
def products_skin():
    return render_template("products_skin.html")

@app.route("/products")
def products():
    return redirect(url_for('products_skin'))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search")
def search():
    return render_template("search.html")

@app.route("/like")
def like():
    return render_template("like.html")

@app.route("/lip_detail")
def lip_detail():
    return render_template("lip_detail.html")

@app.route("/products_eye")
def products_eye():
    return render_template("products_eye.html")

@app.route("/products_lip")
def products_lip():
    return render_template("products_lip.html")

@app.route("/products_jewelry")
def products_jewelry():
    return render_template("products_jewelry.html")

# @app.route("/test")
# def test():
#     return render_template("home.html")


# @app.route("/getlip", methods=["GET","POST"])
# def getlip():
#     return render_template("test2.html")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
