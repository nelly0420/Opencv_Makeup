import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit

import cv2
import numpy as np
from util.lipstick import apply_lipstick
from util.eyeliner import apply_eyeliner
from util.blush import apply_blush
from util.eyebrow import apply_eyebrow
import dlib
import json

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
    makeup_type = request.form.get('type')
    makeup_prdCode = request.form.get('id')
    
    if makeup_type == 'lipstick':
        print('Applying lipstick...')
        #img_with_makeup = apply_lipstick(img)#, prdCode) to-be json의 key값을 가져오기
        #test
        #prdCode = "L00001"
        print(f"Received id: {makeup_prdCode}")
        img_with_makeup = apply_lipstick(img, makeup_prdCode) #to-be json의 key값을 가져오기
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

@app.route("/products_lip_detail")
def lip_detail():
    return render_template("products_lip_detail.html")

@app.route("/products_blush_detail")
def blush_detail():
    return render_template("products_blush_detail.html")

@app.route("/products_eye")
def products_eye():
    return render_template("products_eye.html")

@app.route("/products_lip")
def products_lip():
    return render_template("products_lip.html")

@app.route("/products_jewelry")
def products_jewelry():
    return render_template("products_jewelry.html")


@app.route("/like")
def like():
    return redirect(url_for('like_skin'))

@app.route("/like_skin")
def like_skin():
    return render_template("like_skin.html")

@app.route("/like_eye")
def like_eye():
    return render_template("like_eye.html")

@app.route("/like_lip")
def like_lip():
    return render_template("like_lip.html")

@app.route("/like_jewelry")
def like_jewerly():
    return render_template("like_jewelry.html")

@app.route("/productJSON", methods=["GET"])
def get_products():
    # JSON 파일 경로 설정
    products_file = os.path.join(os.path.dirname(__file__), 'products.json')

    # 파일에서 데이터 읽기
    with open(products_file, 'r', encoding='utf-8') as f:
        products_data = json.load(f)  # Parse JSON data

    # 쿼리 스트링 인자 가져오기 (예: /productJSON?category=lipstick)
    category = request.args.get('category')

    if category:
        # 필터링된 제품 목록 생성
        filtered_products = [product for product in products_data if product['category'] == category]
    else:
        # 카테고리가 지정되지 않은 경우 모든 제품 반환
        filtered_products = products_data

    return jsonify(filtered_products)

@app.route("/productJSON", methods=["POST"])
def aaaaa():
    return "hello"

# 필요하면 사용
# @app.route('/select_product', methods=['GET'])
# def select_product():
#     prd_code = request.args.get('prdCode')
#     # prd_code를 사용하여 원하는 작업을 수행합니다.
#     print(f"Selected product code: {prd_code}")
#     return jsonify({"status": "success", "prdCode": prd_code})
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
