# app.py

from flask import Flask,request
import predict
import cv2
import numpy as np
app = Flask(__name__)

@app.route("/post",methods=['POST'])
def hello():
    data = request.get_json()
    return predict.predict(np.array(data['imgs']))
    