import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import json

from util.lipstick import apply_lipstick
from util.eyeliner import apply_eyeliner
from util.blush import apply_blush
from util.eyebrow import apply_eyebrow2
from util.eyeshadow import apply_eyeshadow
from util.sunglasses import apply_sunglasses
from util.colored_lens import apply_lens

# import dlib
# from datetime import datetime

# Application 정의
app = Flask(__name__, static_url_path="/static") # static 경로 설정이 되어있음.

# Socket 정의
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/sample")
def sample():
    return render_template("sample.html")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route("/queue_display")
def queue_display():
    return render_template("queue_display.html")
    
# @socketio.on('samplelipstick')
# def handle_image_sample(data):
#     # byte ->  numpy array
#     nparr = np.frombuffer(data, np.uint8)
#     # 버퍼에서 이미지 읽기.
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #BGR 형식의 컬러 이미지로 가져오기
#     # img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # 이미지를 원래 품질로 디코딩
#     ##############################################
#     # 아래 부분만 수정.
#     ##############################################
    
    
#     # 립스틱 적용.
#     img_with_makeup = apply_lipstick( img, "LM0001")

#     # 최종적으로 클라이언트로 전송할 때 JPEG로 변환
#     _, buffer = cv2.imencode('.jpg', img_with_makeup, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) #85% 품질.
#     ##################################################
#     # base64로 변환.
#     result_image = base64.b64encode(buffer).decode('utf-8')
#     # Emit 전송.
#     emit('processed_image', {'image': result_image})

# ---------------------------------------------------------------

@socketio.on('send_image')
def handle_image(data):
    # Decode image from bytes
    nparr = np.frombuffer(data['image'], np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    makeup_type = data['category']
    makeup_prdCode = data['prdCode']
    userColor = data['color'] # 사용자 설정 color
    
    if makeup_type == 'lipstick':
        print('Applying lipstick...')
        img_with_makeup = apply_lipstick(img, makeup_prdCode, userColor) #to-be json의 key값을 가져오기(prdCode)
    elif makeup_type == 'eyeliner':
        print('Applying eyeliner...')
        img_with_makeup = apply_eyeliner(img, makeup_prdCode, userColor)
    elif makeup_type == 'blush':
        print('Applying blush...')
        img_with_makeup = apply_blush(img, makeup_prdCode, userColor)
    elif makeup_type == 'eyebrow':
        print('Applying eyebrow...')
        img_with_makeup = apply_eyebrow2(img, makeup_prdCode, userColor)
    elif makeup_type == 'eyeshadow':
        print('Applying eyeshadow...')
        img_with_makeup = apply_eyeshadow(img, makeup_prdCode) # 사용자 설정 color 기능 없음
    elif makeup_type == 'lens':
        print('Applying lens...')
        img_with_makeup = apply_lens(img, makeup_prdCode, userColor) # 사용자 설정 color 기능 없음    
    elif makeup_type == "sunglasses":
        img_with_makeup = apply_sunglasses(img, makeup_prdCode) # 사용자 설정 color 기능 없음
    else:
        print(f'Unknown makeup type: {makeup_type}')
        return
    
    # Encode image back to bytes and send back to client
    _, buffer = cv2.imencode('.jpg', img_with_makeup)
    # base64로 변환.
    img_rslt = base64.b64encode(buffer).decode('utf-8')
    emit('processed_image', {'image': img_rslt})

@app.route("/")
@app.route("/main")
def main():
    return render_template("main.html")

@app.route("/products_skin/<tabname>")
def products_skin(tabname):
    return render_template("products_skin.html", tabname = tabname)

@app.route("/products")
def products():
    return redirect(url_for('products_skin'))
    
@app.route("/about")
def about():
    return render_template("about2.html")

@app.route('/search', methods=['GET'])
def search():
    if 'query' in request.args:
        query = request.args.get('query', '').lower()
        return render_template('search.html', query=query)
    return render_template('search.html')


## ex : /products_blush_detail/B00003
@app.route("/products_blush_detail/", defaults={'prdCode': None})
@app.route("/products_blush_detail/<prdCode>")
def blush_detail(prdCode):
    if prdCode is None:
        return render_template("products_blush_detail.html")
    return render_template("products_blush_detail.html", category='blush', prdCode=prdCode)

@app.route("/products_eyeliner_detail", defaults={'prdCode': None})
@app.route("/products_eyeliner_detail/<prdCode>")
def eyeliner_detail(prdCode):
    if prdCode is None:
        return render_template("products_eyeliner_detail.html")
    return render_template("products_eyeliner_detaill", category='eyeliner', prdCode=prdCode)

@app.route("/products_eyeshadow_detail/", defaults={'prdCode': None})
@app.route("/products_eyeshadow_detail/<prdCode>")
def eyeshadow_detail(prdCode):
    if prdCode is None:
        return render_template("products_eyeshadow_detail.html")
    return render_template("products_eyeshadow_detail.html", category='eyeshadow', prdCode=prdCode) 

@app.route("/products_eyebrow_detail/", defaults={'prdCode': None})
@app.route("/products_eyebrow_detail/<prdCode>")
def eyebrow_detail(prdCode):
    if prdCode is None:
        return render_template("products_eyebrow_detail.html")
    return render_template("products_eyebrow_detail.html", category='eyebrow', prdCode=prdCode)

@app.route("/products_eye")
def products_eye():
    return render_template("products_eye.html")
@app.route("/products_lip")
def products_lip():
    return render_template("products_lip.html")

@app.route("/products_lip_detail_matte/", defaults={'prdCode': None})
@app.route("/products_lip_detail_matte/<prdCode>")
def lip_detail_matte(prdCode):
    if prdCode is None:
        return render_template("products_lip_detail_matte.html")
    return render_template("products_lip_detail_matte.html", category='lipstick', option1='Matte', prdCode=prdCode)


@app.route("/products_lip_detail_glossy/", defaults={'prdCode': None})
@app.route("/products_lip_detail_glossy/<prdCode>")
def lip_detail_glossy(prdCode):
    if prdCode is None:
        return render_template("products_lip_detail_glossy.html")
    return render_template("products_lip_detail_glossy.html", category='lipstick', option1='Glossy', prdCode=prdCode)

@app.route("/products_fashion")
def products_fashion():
    return render_template("products_fashion.html")

@app.route("/products_sunglasses/", defaults={'prdCode': None})
@app.route("/products_sunglasses/<prdCode>")
def products_sunglasses(prdCode):
    if prdCode is None:
        return render_template("products_sunglasses.html")
    return render_template("products_sunglasses.html", category='sunglasses', prdCode=prdCode)

@app.route("/products_lens/", defaults={'prdCode': None})
@app.route("/products_lens/<prdCode>")
def products_lens(prdCode):
    if prdCode is None:
        return render_template("products_lens.html")
    return render_template("products_lens.html", category='lens', prdCode=prdCode)

@app.route("/test")
def test():
    return render_template("test copy.html")

@app.route("/wish")
def wish():
    return render_template("wish.html")

@app.route("/productJSON", methods=["GET"])
def get_products():
    # JSON 파일 경로 설정
    products_file = os.path.join(os.path.dirname(__file__), 'products.json')
    # 파일에서 데이터 읽기
    with open(products_file, 'r', encoding='utf-8') as f:
        products_data = json.load(f)  # Parse JSON data
    # 쿼리 스트링 인자 가져오기
    category = request.args.get('category')
    option1 = request.args.get('option1')
    query = request.args.get('query', '').lower()
    # 필터링된 제품 목록 생성
    filtered_products = products_data
    if category:
        filtered_products = [product for product in filtered_products if product['category'] == category]
    if option1:
        filtered_products = [product for product in filtered_products if product['option1'] == option1]
    if query:
        filtered_products = [product for product in filtered_products if query in product['PrdName'].lower()]
    # else:
    #     # 카테고리가 지정되지 않은 경우 모든 제품 반환
    #     filtered_products = products_data
    return jsonify(filtered_products)


    
@app.route('/apply_color_lens', methods=['POST'])
def apply_color_lens_endpoint():
    if request.method == 'POST':
        # Get the FormData object from the request
        data = request.form
        # Extract prdCode from the FormData
        prd_code = data.get('prdCode')
        # Assuming you also want to receive the image data
        image_data = request.files['image'].read()
        nparr = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Apply blush effect to the image
        result_image = apply_eyeshadow(image_np, prd_code)
        # Encode the processed image to JPEG format
        _, buffer = cv2.imencode('.jpg', result_image)
        # Return the processed image as bytes with a status code
        return buffer.tobytes(), 200
    else:
        return "Method not allowed", 405

#안전하지 않은 방법으로 werkzeug를 실행하는 경우 경고를 표시하거나 실행을 차단할 수 있습니다. 이때 allow_unsafe_werkzeug=True 옵션을 설정하면 이러한 경고를 무시하고 애플리케이션을 실행할 수 있게 해줍니다.
if __name__ == '__main__':
    # socketio.run(app, port=5000, debug=True, host='0.0.0.0', allow_unsafe_werkzeug=True)
    socketio.run(app, port=5000, debug=True, host='0.0.0.0')