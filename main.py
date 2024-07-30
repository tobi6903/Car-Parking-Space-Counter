
import cv2
import numpy as np
import cvzone
import pickle
width,height = (158-50), (240-192)
cap=cv2.VideoCapture("carPark.mp4")

with open('CarParkPos','rb') as f:
    posList=pickle.load(f)

def checkParkingSpace(imgPro):
    spaceCounter = 0
    for pos in posList:
        x,y=pos
        # cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
        imgCrop=imgPro[y:y+height,x:x+width]
        cv2.imshow(str(x*y),imgCrop)
        count=cv2.countNonZero(imgCrop)
        cvzone.putTextRect(img,str(count),(x,y+height-2),scale=1,offset=1,thickness=2,colorR=(0,0,255))
        if(count<500):
            color = (0,255,0)
            thickness=5
            spaceCounter+=1
        else :
            color=(0,0,255)
            thickness=2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
    cvzone.putTextRect(img,f'Free Space{str(spaceCounter)}/{len(posList)}',(400,50),scale=4,thickness=2,offset=10,colorR=(0,200,0))

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()

    imageGray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    imgBlur=cv2.GaussianBlur(imageGray,(3,3),1)
    imgThreshold=cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY_INV,25,16) #gets converted to binary image
    imgMedian=cv2.medianBlur(imgThreshold,5)
    kernel=np.zeros((3,3),np.uint8);
    imgDialate=cv2.dilate(imgMedian,kernel,iterations=1) #edges enlarge

    checkParkingSpace(imgDialate)

    cv2.imshow('ImageThres', imgThreshold)
    cv2.imshow('ImageMed', imgMedian)
    cv2.imshow('ImageGaussinaBlur', imgBlur)
    cv2.imshow('ImageGra', imageGray)
    cv2.imshow('ImageDia', imgDialate)
    cv2.imshow('Image', img)
    cv2.waitKey(1)