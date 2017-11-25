import cv2
import numpy as np
import os
import time
import uuid
import math
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# The cutoff for threshold. A lower number means smaller changes between
# the average and current scene are more readily detected.
THRESHOLD_SENSITIVITY = 20 # default value = 50
# Number of pixels in each direction to blur the difference between
# average and current scene. This helps make small differences larger
# and more detectable.
BLUR_SIZE = 40 
# The number of square pixels a blob must be before we consider it a
# candidate for tracking.
BLOB_SIZE = 500 # default = 500
# The number of pixels wide a blob must be before we consider it a
# candidate for tracking.
BLOB_WIDTH = 150 # default = 60
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
DEFAULT_AVERAGE_WEIGHT = 0.04 # default = 0.04
# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DISTANCE_PX = 80 # default = 80
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7
# Constants for drawing on the frame.
LINE_THICKNESS = 1
CIRCLE_SIZE = 5
RESIZE_RATIO = 0.4 # default = 0.4

# switch camera to video streaming
vc		 = cv2.VideoCapture("videos/session2_left.avi")
# vc = cv2.Videovcture(1)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
# With webcam get(CV_vc_PROP_FPS) does not work.
# Let's see for ourselves.
 
if int(major_ver) < 3 :
	fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
	# get vcap property 
	width = vc.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
	height = vc.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float
else :
	fps = vc.get(cv2.CAP_PROP_FPS)
	# get vcap property 
	width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
	height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def func_kalman_xy(dict_xy):
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty
    observed_x = []
    observed_y = []
    result_array = []
    for item in dict_xy:
        observed_x.append(item[0])
        observed_y.append(item[1])
    N = 20
    result = []
    R = 0.01**2
    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    for i in range(len(kalman_x)):
        pass
        item_a = (round(kalman_x[i][0]), round(kalman_y[i][0]))
        result_array.append(item_a)
    return result_array

def nothing(*args, **kwargs):
	" A helper function to use for OpenCV slider windows. "
	print (args, kwargs)

def calculate_speed (trails, fps):
	# distance: distance on the frame
	# location: x, y coordinates on the frame
	# fps: framerate
	# mmp: meter per pixel
	dist = cv2.norm(trails[0], trails[10])
	dist_x = trails[0][0] - trails[10][0]
	dist_y = trails[0][1] - trails[10][1]

	mmp_y = 0.2 / (3 * (1 + (3.22 / 432)) * trails[0][1])
	mmp_x = 0.2 / (5 * (1 + (1.5 / 773)) * (width - trails[0][1]))
	real_dist = math.sqrt(dist_x * mmp_x * dist_x * mmp_x + dist_y * mmp_y * dist_y * mmp_y)

	return real_dist * fps * 250 / 3.6

def get_frame():
	" Grabs a frame from the video vcture and resizes it. "
	rval, frame = vc.read()
	if rval:
		(h, w) = frame.shape[:2]
		frame = cv2.resize(frame, (int(w * RESIZE_RATIO), int(h * RESIZE_RATIO)), interpolation=cv2.INTER_CUBIC)
	return rval, frame

from itertools import *
def pairwise(iterable):
	"s -> (s0,s1), (s1,s2), (s2, s3), ..."
	a, b = tee(iterable)
	next(b, None)
	return zip(a, b)

# cv2.namedWindow("preview")
# cv2.cv.SetMouseCallback("preview", nothing)

# A variable to store the running average.
avg = None
# A list of "tracked blobs".
tracked_blobs = []

a = []
model_dir = ''
bgsMOG = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold = 50, detectShadows=0)
if vc:
	while True:
		# Grab the next frame from the camera or video file
		grabbed, frame = get_frame()

		if not grabbed:
			# If we fall into here it's because we ran out of frames
			# in the video file.
			break

		frame_time = time.time()
		if grabbed:
			fgmask = bgsMOG.apply(frame, None, 0.01)
			# To find the contours of the objects
			_, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# cv2.drawContours(frame,contours,-1,(0,255,0),cv2.cv.CV_FILLED,32)
			try: hierarchy = hierarchy[0]
			except: hierarchy = []
			a = []
			for contour, hier in zip(contours, hierarchy):
				(x, y, w, h) = cv2.boundingRect(contour)

				if w < 80 and h < 80:
					continue

				center = (int(x + w/2), int(y + h/2))

				if center[1] > 320 or center[1] < 150:
					continue

				# Optionally draw the rectangle around the blob on the frame that we'll show in a UI later
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				# Look for existing blobs that match this one
				closest_blob = None
				if tracked_blobs:
					# Sort the blobs we have seen in previous frames by pixel distance from this one
					closest_blobs = sorted(tracked_blobs, key=lambda b: cv2.norm(b['trail'][0], center))

					# Starting from the closest blob, make sure the blob in question is in the expected direction
					distance = 0.0
					distance_five = 0.0
					for close_blob in closest_blobs:
						distance = cv2.norm(center, close_blob['trail'][0])
						if len(close_blob['trail']) > 10:
							distance_five = cv2.norm(center, close_blob['trail'][10])
						
						# Check if the distance is close enough to "lock on"
						if distance < BLOB_LOCKON_DISTANCE_PX:
							# If it's close enough, make sure the blob was moving in the expected direction
							expected_dir = close_blob['dir']
							if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
								continue
							elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
								continue
							else:
								closest_blob = close_blob
								break

					if closest_blob:
						# If we found a blob to attach this blob to, we should
						# do some math to help us with speed detection
						prev_center = closest_blob['trail'][0]
						if center[0] < prev_center[0]:
							# It's moving left
							closest_blob['dir'] = 'left'
							closest_blob['bumper_x'] = x
						else:
							# It's moving right
							closest_blob['dir'] = 'right'
							closest_blob['bumper_x'] = x + w

						# ...and we should add this centroid to the trail of
						# points that make up this blob's history.
						closest_blob['trail'].insert(0, center)
						closest_blob['last_seen'] = frame_time
						if len(closest_blob['trail']) > 10:
							closest_blob['speed'].insert(0, calculate_speed (closest_blob['trail'], fps))

				if not closest_blob:
					# If we didn't find a blob, let's make a new one and add it to the list
					b = dict(
						id=str(uuid.uuid4())[:8],
						first_seen=frame_time,
						last_seen=frame_time,
						dir=None,
						bumper_x=None,
						trail=[center],
						speed=[0],
						size=[0, 0],
					)
					tracked_blobs.append(b)

			cv2.imshow('BGS', fgmask)

		if tracked_blobs:
			# Prune out the blobs that haven't been seen in some amount of time
			for i in range(len(tracked_blobs) - 1, -1, -1):
				if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
					print ("Removing expired track {}".format(tracked_blobs[i]['id']))
					del tracked_blobs[i]

		# Draw information about the blobs on the screen
		print ('tracked_blobs', tracked_blobs)
		for blob in tracked_blobs:
			for (a, b) in pairwise(blob['trail']):
				cv2.circle(frame, a, 3, (255, 0, 0), LINE_THICKNESS)

				# print ('blob', blob)
				if blob['dir'] == 'left':
					pass
					cv2.line(frame, a, b, (255, 255, 0), LINE_THICKNESS)
				else:
					pass
					cv2.line(frame, a, b, (0, 255, 255), LINE_THICKNESS)

				# bumper_x = blob['bumper_x']
				# if bumper_x:
				#	 cv2.line(frame, (bumper_x, 100), (bumper_x, 500), (255, 0, 255), 3)
				# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), LINE_THICKNESS)
				# cv2.circle(frame, center, 10, (0, 255, 0), LINE_THICKNESS)

			if blob['speed'] and blob['speed'][0] != 0:

				# remove zero elements on the speed list
				blob['speed'] = [item for item in blob['speed'] if item != 0.0]
				print ('========= speed list =========', blob['speed'])
				ave_speed = np.mean(blob['speed'])
				print ('========= ave_speed =========', ave_speed)
				cv2.putText(frame, str(int(ave_speed)) + 'km/h', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)

		print ('*********************************************************************')
		# Show the image from the camera (along with all the lines and annotations)
		# in a window on the user's screen.
		cv2.imshow("BGS Method", frame)

		key = cv2.waitKey(10)
		if key == 27: # exit on ESC
			break