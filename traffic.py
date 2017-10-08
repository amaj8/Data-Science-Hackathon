import numpy as np
import cv2
import matplotlib.pyplot as plt

thresh=2
fps = 24
plt.axis([0, 10, 0, 6])


plt.ion()
# plt.ioff()

cascade_src = 'cars.xml'

video_src = 'video1.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
l = None
i = 1
frame_counter = 1
while True:
	ret, img = cap.read()
	if (type(img) == type(None)):
		break

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cars = car_cascade.detectMultiScale(gray, 1.1, 1)
	ls=[]

	# i = 1
	prev=1
	l1=2
	for (x,y,w,h) in cars:
		global l
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		l=len(cars)
		#ls.append(l)
		i += 1
		# print "i="+ str(i)
		if i==prev+125:
			if l>=l1+thresh:
				time = i/fps

				print "traffic Jam detected at time = "+ str(time) + "s" 
				l1=l
			else :
				print "No traffic jam"
			prev=i


		#plt.plot(ls)
		#plt.ylabel('count')
		if(i%3==0):
			plt.ylabel('no of cars')
			plt.xlabel('Frame')
			plt.scatter(i, l)


	# plt.show()

	cv2.imshow('video', img)

	if cv2.waitKey(33) == 27:
		break

	plt.pause(0.05)















