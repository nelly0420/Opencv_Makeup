import os
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask import Flask, redirect, url_for

import cv2
import numpy as np
from util.lipstick import apply_lipstick
from util.eyeliner import apply_eyeliner
from util.blush import apply_blush
from util.eyebrow import apply_eyebrow
import dlib

# Application 정의
app = Flask(__name__, static_url_path="/static")

# Socket 정의
socketio = SocketIO(app, cors_allowed_origins="*")

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
    nparr = np.frombuffer(data['image'], np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply makeup based on the selected option
    makeup_type = data.get('type')
    if makeup_type == 'lipstick':
        print('Applying lipstick...')
        img_with_makeup = apply_lipstick(img)
    elif makeup_type == 'eyeliner':
        print('Applying eyeliner...')
        img_with_makeup = apply_eyeliner(img)
    elif makeup_type == 'blush':
        print('Applying blush...')
        img_with_makeup = apply_blush(img)
    elif makeup_type == 'eyebrow':
        print('Applying eyebrow...')
        img_with_makeup = apply_eyebrow(img)
    else:
        print(f'Unknown makeup type: {makeup_type}')
        return

    # Encode image back to bytes and send back to client
    _, buffer = cv2.imencode('.jpg', img_with_makeup)
    emit('processed_image', {'image': buffer.tobytes()})

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

@app.route("/products_lip_detail")
def lip_detail():
    return render_template("products_lip_detail.html")

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

@app.route("/productJSON" , methods=["GET"] )
def test():
    products_file = os.path.join(os.path.dirname(__file__), 'products.json')
    with open(products_file, 'r', encoding='utf-8') as f:
        products = f.read() #파일 전체 
        # (필터링)...
        # if lip 에 대해서 내려주면 된다. ................
        # flask get 방식으로 넘길때 인자값을 어떻게 넘기는 지?
    return jsonify(products) 

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
