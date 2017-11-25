import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# switch camera to video streaming
cap		 = cv2.VideoCapture("output.mp4") # videos/session0_left.avi
# cap = cv2.VideoCapture(1)

a = []
model_dir = ''
bgsMOG = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold = 50, detectShadows=0)
if cap:
	while True:
		ret, frame = cap.read()
		if ret:
			fgmask = bgsMOG.apply(frame, None, 0.01)
			# To find the contours of the objects
			_, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# cv2.drawContours(frame,contours,-1,(0,255,0),cv2.cv.CV_FILLED,32)
			try: hierarchy = hierarchy[0]
			except: hierarchy = []
			a = []
			for contour, hier in zip(contours, hierarchy):
				(x, y, w, h) = cv2.boundingRect(contour)

				if w > 30 and h > 30:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					(x, y, w, h) = cv2.boundingRect(contour)

					x1 = w / 2
					y1 = h / 2
					cx = x + x1
					cy = y + y1
					a.append([cx, cy])
					# print(len(a))

			cv2.imshow('BGS', fgmask)
			cv2.imshow('Ori+Bounding Box', frame)
			key = cv2.waitKey(100)
			if key == ord('q'):
				break
cap.release()
cv2.destroyAllWindows()