import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit # framework -> Flask, Django, FastAPI
import cv2
import numpy as np
from util.lipstick import apply_lipstick, apply_lipstick2
from util.eyeliner import apply_eyeliner
from util.blush import apply_blush
from util.eyebrow import apply_eyebrow
from util.eyeshadow import apply_eyeshadow
import dlib
import json
import base64
from datetime import datetime

# Application 정의
app = Flask(__name__, static_url_path="/static") # static 경로 설정이 되어있음.
# Socket 정의
socketio = SocketIO(app, cors_allowed_origins="*")
# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

@app.route("/sample")
def sample():
    return render_template("sample.html")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('samplegray')
def handle_image_sample(data):
    # byte ->  numpy array
    nparr = np.frombuffer(data, np.uint8)
    # 버퍼에서 이미지 읽기.
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # 이미지를 원래 품질로 디코딩
    ##############################################
    # Convert image to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 립스틱 적용.
    img_with_makeup = apply_lipstick2(img, "LM0001")

    # 최종적으로 클라이언트로 전송할 때 JPEG로 변환
    _, buffer = cv2.imencode('.jpg', img_with_makeup, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) #85% 품질.
    ##################################################
    # base64로 변환.
    result_image = base64.b64encode(buffer).decode('utf-8')
    # Emit 전송.
    emit('processed_image', {'image': result_image})

# ---------------------------------------------------------------

# @socketio.on('image')
# def handle_image(data):
#     # Decode image from bytes
#     nparr = np.frombuffer(data['image'], np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     # Apply makeup based on the selected option
#     makeup_type = data.get('type')
#     makeup_prdCode = data.get('id')
#     if makeup_type == 'lipstick':
#         print('Applying lipstick...')
#         img_with_makeup = apply_lipstick(img, makeup_prdCode) #to-be json의 key값을 가져오기
#     elif makeup_type == 'eyeliner':
#         print('Applying eyeliner...')
#         img_with_makeup = apply_eyeliner(img, makeup_prdCode)
#     elif makeup_type == 'blush':
#         print('Applying blush...')
#         img_with_makeup = apply_blush(img, makeup_prdCode)
#     elif makeup_type == 'eyebrow':
#         print('Applying eyebrow...')
#         img_with_makeup = apply_eyebrow(img, makeup_prdCode)
#     elif makeup_type == 'eyeshadow':
#         print('Applying eyeshadow...')
#         img_with_makeup = apply_eyeshadow(img, makeup_prdCode)
#     else:
#         print(f'Unknown makeup type: {makeup_type}')
#         return
#     # Encode image back to bytes and send back to client
#     _, buffer = cv2.imencode('.jpg', img_with_makeup)
#     # base64로 변환.
#     img_rslt = base64.b64encode(buffer).decode('utf-8')
#     emit('processed_image', {'image': img_rslt})

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
    return render_template("products_lip_detail_matte.html", category='lipstick', prdCode=prdCode)


@app.route("/products_lip_detail_glossy/", defaults={'prdCode': None})
@app.route("/products_lip_detail_glossy/<prdCode>")
def lip_detail_glossy(prdCode):
    if prdCode is None:
        return render_template("products_lip_detail_glossy.html")
    return render_template("products_lip_detail_glossy.html", category='lipstick', prdCode=prdCode)

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
# @app.route("/productJSON", methods=["POST"])
# def aaaaa():
#     return "hello"
# @app.route("/productJSON", methods=["POST"])
# def aaaaa():
#     if request.method == 'POST':
#             data = request.form  # 폼 데이터를 가져옴
#             print(data)  # 콘솔에 출력하여 확인
#             # 필요한 처리 로직 수행
#             return 'Success'
# @app.route("/productJSON", methods=["GET"])
# def get_product_json():
#     prd_code = request.args.get('prdCode')
#     # prdCode를 사용하여 필요한 로직 수행
#     print(f'Product Code: {prd_code}')
#     # 필요한 처리 로직 수행
#     return jsonify({'status': 'Success', 'prdCode': prd_code})
@app.route('/apply_blush', methods=['POST'])
def apply_blush_endpoint():
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
        result_image = apply_blush(image_np, prd_code)
        # Encode the processed image to JPEG format
        _, buffer = cv2.imencode('.jpg', result_image)
        # Return the processed image as bytes with a status code
        return buffer.tobytes(), 200
    else:
        return "Method not allowed", 405
@app.route('/apply_lipstick', methods=['POST'])
def apply_lip_endpoint():
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
        result_image = apply_lipstick(image_np, prd_code)
        # Encode the processed image to JPEG format
        _, buffer = cv2.imencode('.jpg', result_image)
        # Return the processed image as bytes with a status code
        return buffer.tobytes(), 200
    else:
        return "Method not allowed", 405
@app.route('/apply_eyeliner', methods=['POST'])
def apply_eyeliner_endpoint():
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
        result_image = apply_eyeliner(image_np, prd_code)
        # Encode the processed image to JPEG format
        _, buffer = cv2.imencode('.jpg', result_image)
        # Return the processed image as bytes with a status code
        return buffer.tobytes(), 200
    else:
        return "Method not allowed", 405
    
@app.route('/apply_eyeshadow', methods=['POST'])
def apply_eyeshadow_endpoint():
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
    
@app.route('/apply_eyebrow', methods=['POST'])
def apply_eyebrow_endpoint():
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
        result_image = apply_eyebrow(image_np, prd_code)
        # Encode the processed image to JPEG format
        _, buffer = cv2.imencode('.jpg', result_image)
        # Return the processed image as bytes with a status code
        return buffer.tobytes(), 200
    else:
        return "Method not allowed", 405
    
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)