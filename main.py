# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:23:38 2020

@author: fonte
"""
import matplotlib.pyplot as plt
import os
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from object_detection import ObjectDetection
import tempfile
import copy

import cv2
#import sys
#from mail import sendEmail
from flask import Flask, render_template, Response
from webcamvideostream import WebcamVideoStream
from flask_basicauth import BasicAuth
#import time
#import threading

##############################################################################
MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def predict_image(frame, od_model):
#    image = Image.open(image_filename)
    image = frame#cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)
    predictions = od_model.predict_image(Image.fromarray(image))
    
    img_result = copy.deepcopy(image) # faz copia da imagem
    dado = list()
    for i in range(len(predictions)):
        if predictions[i]['probability'] > 0.4:
            dado.append(predictions[i])
            lef = predictions[i]['boundingBox']['left']
            top = predictions[i]['boundingBox']['top']
            wid = predictions[i]['boundingBox']['width']
            hei = predictions[i]['boundingBox']['height']
            A = int(lef*img_result.shape[1])
            B = int(top*img_result.shape[0])
            C = int((lef+wid)*img_result.shape[1])
            D = int((top+hei)*img_result.shape[0])
            E = B-10	
            img_result = cv2.rectangle(img_result, (A, B), (C, D), (255, 0, 255), 2)

            text = str(np.round(100*predictions[0]['probability'],1)) + ' %'
            img_result = cv2.putText(img_result,
                                     text,
                                     (A, E),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.4,
                                     (255, 0, 255),
                                     1,
                                     cv2.LINE_AA)
            text2 = 'Com Mascara'  
            img_result = cv2.putText(img_result,
                                     text2,
                                     (5, 5),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,
                                     (255, 0, 255),
                                     1,
                                     cv2.LINE_AA)
        else:
            text3 = 'Sem Mascara'  
            img_result = cv2.putText(img_result,
                                     text3,
                                     (5, 5),
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     1,
                                     (0, 255, 255),
                                     1,
                                     cv2.LINE_AA)

    return dado, img_result
###############################################################################
    
#email_update_interval = 600 # sends an email only once in this time interval
#video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
#object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier

# App Globals (do not edit)
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'pi'
app.config['BASIC_AUTH_PASSWORD'] = 'pi'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)
last_epoch = 0

# Load labels
with open(LABELS_FILENAME, 'r') as f:
     labels = [l.strip() for l in f.readlines()]

od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)


@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        if camera.stopped:
            break
        frame = camera.read()
        dado, img_result = predict_image(frame, od_model)
        ret, jpeg = cv2.imencode('.jpg',img_result)
#        ret, jpeg = cv2.imencode('.jpg',frame)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            print("frame is none")

@app.route('/video_feed')
def video_feed():
    dado, img_result = predict_image(frame, od_model)
    return Response(gen(WebcamVideoStream().start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True, threaded=True)
    app.run()