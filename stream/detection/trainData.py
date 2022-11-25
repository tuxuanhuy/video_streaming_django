import cv2
import numpy as np
import os
# thu vien de co the trich xuat duoc anh tu thu muc
from PIL import Image

# import thu vien mac dinh cua opencv de train cho nhan dien hinh anh
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'facedata/'

# lay anh tu folder
def getImagefromId(path):
     
     # lay duong dan cua anh tu folder, os truy cap vao duong dan, os.listdir(path) truy cap vao tat ca cac file trong facedata
    imgPaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imgPath in imgPaths:
        # lay tat ca cac anh trong folder va chuyen doi sang kieu anh pil de thao tac voi mang de train
        faceImg = Image.open(imgPath).convert('L')
        
        # convert faceImg ve kieu array de su dung de train
        faceNp = np.array(faceImg, 'uint8')
        
        # tach lay ID
        Id = int(imgPath.split('/')[1].split('.')[1])

        faces.append(faceNp)
        IDs.append(Id)

        cv2.imshow('training', faceNp)
        cv2.waitKey(10)

    return faces, IDs


faces, Ids = getImagefromId(path)

# su dung recognizer de train
recognizer.train(faces, np.array(Ids))

# tao folder de luu du lieu da train
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

# luu giu lieu da train vao file yml
recognizer.save('recognizer/trainData.yml')

cv2.destroyAllWindows()