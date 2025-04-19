from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import uuid

app = Flask(__name__)
model = YOLO("yolov5s.pt")  # استخدم الموديل المتوفر لديك

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = str(uuid.uuid4()) + "_" + file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # تحليل الصورة باستخدام YOLO
    results = model(path)
    results[0].save(filename=path)  # يحفظ الصورة مع المربعات فوقها بنفس الاسم

    return render_template('index.html', image_path=path)

if __name__ == '__main__':
    app.run(debug=True)
