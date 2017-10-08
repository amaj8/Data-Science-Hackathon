import numpy as np
import cv2
import matplotlib.pyplot as plt




cascade_src = 'pedestrian.xml'

video_src = 'video3.mp4'

cap = cv2.VideoCapture(video_src)
ped_cascade = cv2.CascadeClassifier(cascade_src)


while True:
	ret, img = cap.read()
	if (type(img) == type(None)):
		break

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	body = ped_cascade.detectMultiScale(gray, 1.1, 1)

	for (x,y,w,h) in body:
		global l
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		print "Current no. of pedestrians detected = " + str(len(body))
	

	cv2.imshow('video', img)

	if cv2.waitKey(2) == 27:
		break