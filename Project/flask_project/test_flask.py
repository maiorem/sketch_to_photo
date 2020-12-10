import flask
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import imageio
import sys
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import matplotlib.pyplot as plt
import cv2
import pandas as pd


app=Flask(__name__, static_url_path="/static", static_folder="static")

# 메인페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

# 데이터 예측 처리
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method=='POST' :

        file=request.files['img']
        if not file : return render_template('index.html', label="NoFiles")

        img=imageio.imread(file)
        img=cv2.resize(img, (256, 256))
        img=img_to_array(img)
        img = (img - 127.5) / 127.5
        img = expand_dims(img, 0)
        print(img.shape)

        model = load_model('./flask_project/model_087200.h5')
        predict=model.predict(img)

        predict = (predict + 1) / 2.0
        # plot the image
        plt.imshow(predict[0])
        plt.axis('off')
        plt.savefig('./flask_project/static/out3.png')
        return render_template("index.html", fake_img='out3.png', name=predict[0].shape)

  




if __name__ == '__main__' :
    app.run(host="127.0.0.1", port="8080")