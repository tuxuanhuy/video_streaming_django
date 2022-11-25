import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# training hinh ảnh va thu vien nhan dien
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('detection\\recognizer\\trainData.yml')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # lay profile bang id tu data
    def getProfile(id):

        conn = sqlite3.connect('D:\Final\AIdemo\AiDemo.db')
        query = "SELECT * FROM user WHERE id = " + str(id)
        rs = conn.execute(query)

        profile = None

        for row in rs:
            profile = row

        conn.close()
        return profile

    def get_frame(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        _, image = self.video.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for(x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # cat anh 
            grayImg = gray[y: y+h, x: x+w]

            # nhan dien khuon mat dang trong webcam la ai, neu nguoi do co trong tap giu lieu train thi se tra ve Id va do sai lech
            id, confidence = recognizer.predict(grayImg)

            if confidence < 40:
                profile = self.getProfile(id)

                if(profile != None):
                    # in ra thong tin ve profile
                    cv2.putText(image, "Name: "+str(profile[1]), (x+ 10, y+h+ 30), font, 1, (0, 225, 0), 2)
                    cv2.putText(image, "Age: "+str(profile[2]), (x+ 10, y+h+ 60), font, 1, (0, 225, 0), 2)
                    cv2.putText(image, "Gender: "+str(profile[3]), (x+ 10, y+h+ 90), font, 1, (0, 225, 0), 2)
            else:
                cv2.putText(image, "Unknown", (x+ 10, y+h+ 30), font, 1, (0, 0, 225), 2)

        frame_flip = cv2.flip(image, 1)

        ret, jpeg = cv2.imencode('.jpg', frame_flip)

        return jpeg.tobytes()

            
            
            
            

            
