import cv2
import numpy as np
import sqlite3
# truy cap vao he thong de lay duoc duong dan cac thu muc trong may
import os

def insertOrUpdate(id, name, age, gender):

	conn = sqlite3.connect('D:\Final\AIdemo\AiDemo.db')

	query = "SELECT * FROM user WHERE id = " + str(id)
	rs = conn.execute(query)

	isExist = 0

	for row in rs:
		isExist = 1

	if(isExist == 0):
		query = "INSERT INTO user(id, name, age, gender) VALUES(" +str(id)+ ", '"+str(name)+"', "+str(age)+", '"+str(gender)+"')"
	else:
		query = "UPDATE user SET name = '" + str(name)+ "' ,age = " + str(age) + ", gender = '" + str(gender)+ "' WHERE id =" + str(id)

	conn.execute(query)
	conn.commit()
	conn.close()

# load thu vien
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

# insert vao db
id = input("Nhap id: ")
name = input("Nhap ten: ")
age = input("Nhap tuoi: ")
gender = input("Nhap gioi tinh: ")

insertOrUpdate(id, name, age, gender)

num = 0

while(True):

	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

	for(x, y, w, h) in faces:

		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 2)
        
        # tao folder luu anh vua cat duoc tu hinh vuong
		if not os.path.exists('facedata'):
			os.makedirs('facedata')

		num += 1
        
        # lu anh vao folder facedata
		cv2.imwrite('facedata/User.'+str(id)+'.'+str(num)+'.jpg', gray[y: y+h, x: x+w])

	cv2.imshow('frame', frame)
	cv2.waitKey(1)

	if num > 100:
		break

cap.release()
cv2.destroyAllWindows()

exec(open('trainData.py').read())