# -*- coding: utf-8 -*-
# time: 2023/4/26 10:28
# file: app.py
# 公众号: 玩转测试开发
from flask import Flask, render_template, Response
import cv2

from threading import Thread

from time import sleep

import os




app = Flask(__name__)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
print(f"Current process PID: {os.getpid()}")
app.run(debug=True, port=7998, host="0.0.0.0")

