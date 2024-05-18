import base64
from flask import jsonify
import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.h5")
font = cv2.FONT_HERSHEY_DUPLEX

class VideoCamera(object):
    def __init__(self):
        camera_index = 0
        self.video = cv2.VideoCapture(camera_index)

        if not self.video.isOpened():
            print(f"Failed to open camera at index {camera_index}")
        else:
            print(f"Camera at index {camera_index} opened successfully.")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict_emotion(cropped_img)
            cv2.putText(frame, prediction, (x+20, y-60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', cv2.resize(frame,(800,480), interpolation = cv2.INTER_CUBIC))
        return jpeg.tobytes()

def predict_emotion(image_data):
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict_emotion(cropped_img)
        cv2.putText(image, prediction, (x+20, y-60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    _, jpeg = cv2.imencode('.jpg', cv2.resize(image,(800,480), interpolation = cv2.INTER_CUBIC))

    processed_image_base64 = base64.b64encode(jpeg).decode('utf-8')

    return jsonify({ "processed-image": processed_image_base64 })