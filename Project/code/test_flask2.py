import flask
import sys
from tensorflow.keras.models import load_model
from flask import Flask
from flask import request
from flask import send_from_directory
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import requests
import json


##########모델 로드

model = load_model('model_002760.h5')

##########모델 예측

app = Flask(__name__)

@app.route('/index') #http://127.0.0.1:8000/
def index():
    return flask.render_template("index2.html")

@app.route('/predict', methods=['post']) #http://127.0.0.1:8000/predict
def predict():
    image = Image.open(request.files['file'].stream)
    image = image.convert('RGB') 
    image_numpy = np.array(image) 
    x_test = np.array([image_numpy])

    res = requests.post('http://127.0.0.1:8000/predict', data=json.dumps({'inputs': {'input_tensor': x_test.tolist()}}))

    text = res.text
    #print(text)
    output_dict = json.loads(text)['outputs']
    detection_boxes = output_dict['detection_boxes']
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']
    num_detections = output_dict['num_detections']

    draw = ImageDraw.Draw(image)
    for i in range(num_detections[0]):
        detection_box = detection_boxes[0][i]
        detection_class = detection_classes[0][i]
        detection_score = detection_scores[0][i]

        y1, x1, y2, x2 = detection_box
        width, height = image.size
        y1, x1, y2, x2 = y1 * height, x1 * width, y2 * height, x2 * width
        name = model[detection_class]['name']
        probability = detection_score
        draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 255), width=2)

        if probability < 0.3:
          continue

        text = '{} {:.2f}%'.format(name, probability * 100)  
        text_width, text_height = ImageFont.load_default().getsize(text)
        draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill=(0, 0, 255))    
        draw.text((x1, y1 - text_height), text, font=ImageFont.load_default(), fill=(255, 255, 255))

    image.save('out.png')

    return '<img src="/result/out.png">'

@app.route('/result/<file_name>') #http://127.0.0.1:5000/result/out.png
def result(file_name):
    return send_from_directory(".", file_name)

app.run(host='127.0.0.1', port=5000, debug=False)