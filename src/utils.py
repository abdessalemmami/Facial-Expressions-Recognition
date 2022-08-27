from PIL import Image
import numpy as np
import logging
#import time
import math
import cv2
import os 
import tensorflow as tf

################# LOGGER
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[1;33m"
    red = "\x1b[31;20m"
    green = "\x1b[1;32m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[1m"
    reset = "\x1b[0m"
    #format = time.asctime(time.localtime()) + " - [%(levelname)s] - %(message)s"
    level = "[%(levelname)s] " + reset
    message = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + level + white + message + reset,
        logging.INFO: green + level + white + message + reset,
        logging.WARNING: yellow + level + white + message + reset,
        logging.ERROR: bold_red + level + white + message + reset,
        logging.CRITICAL: red + level + white + message + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger with 
logger = logging.getLogger("FERv2")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

################# generate.py utils

def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))


def str_to_image(image_blob):
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)


#################### processing.py

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, left_eye, right_eye):

	#this function aligns given face in img based on left and right eye coordinates

	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye

	#-----------------------
	#find rotation direction

	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#-----------------------
	#find length of triangle edges

	a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

	#-----------------------

	#apply cosine rule

	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree

		#-----------------------
		#rotate base image

		if direction == -1:
			angle = 90 - angle

		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))

	#-----------------------

	return img #return img anyway



def detect_face(face_detector, img, align = True):
	resp = []
	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
	detections = face_detector.detect_faces(img_rgb)

	if len(detections) > 0:

		for detection in detections:
			x, y, w, h = detection["box"]
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			img_region = [x, y, w, h]

			if align:
				keypoints = detection["keypoints"]
				left_eye = keypoints["left_eye"]
				right_eye = keypoints["right_eye"]
				detected_face = alignment_procedure(detected_face, left_eye, right_eye)

			resp.append((detected_face, img_region))

	return resp



def load_image(img):
    if os.path.isfile(img) != True:
        raise ValueError(img," doesn't exist.")
    return cv2.imread(img,0)







