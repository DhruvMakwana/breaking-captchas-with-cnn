#import libraries
#to run this program run following command in command line
#python test_model.py --input downloads --model output/lenet.hdf5
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

#argument parser for command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, 
	help="path to input directory of images")
#--dataset is the path to the input captcha images that we wish to break
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
#--model is the  path to the serialized weights residing on disk
args = vars(ap.parse_args())

print("[Info] loading pre-trained network...")
model = load_model(args["model"])

#randomly sample a few of input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

#loop over the image paths
for imagePath in imagePaths:
	#load the image and convert it to grayscale, then pad the image
	#to ensure digits caught only the border of the image are
	#retained
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
	#cv2.copyMakeBorder() method is used to create a border around 
	#the image like a photo frame.
	#cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
	#The row or column at the very edge of the original is replicated 
	#to the extra border.

	#threshold the image to reveal the digits
	thresh = cv2.threshold(gray, 0, 255, 
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#Thresholding is a technique in OpenCV, which is the assignment
	#of pixel values in relation to the threshold value provided. 
	#In thresholding, each pixel value is compared with the 
	#threshold value. If the pixel value is smaller than the 
	#threshold, it is set to 0, otherwise, it is set to a maximum 
	#value (generally 255).
	#cv2.thresholding(src, thresholdValue, maxVal, thresholdingTechnique)

	#find contours in the image, keeping only the four largest ones,
	#then sort them from left-to-right
	cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=False)[:4]
	cnts = contours.sort_contours(cnts)[0]
	#Contours are defined as the line joining all the points along 
	#the boundary of an image that are having the same intensity. 
	#the third digit we wish to recognize may be first in the cnts list. 
	#Since we read digits from left-to-right, we need to sort the 
	#contours from left-to-right. This is accomplished via the 
	#sort_contours function 

	#initialize the output image as a "grayscale" image with 3
	#channels along with the output predictions
	output = cv2.merge([gray] * 3)
	predictions = []

	#loop over the contours
	for c in cnts:
		#compute the bounding box for the contour then extract the
		#digit
		(x, y, w, h) = cv2.boundingRect(c)
		roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

		#pre-process the ROI and classify it then classify it
		roi = preprocess(roi, 28, 28)
		roi = np.expand_dims(img_to_array(roi), axis=0)
		pred = model.predict(roi).argmax(axis=1)[0] + 1
		predictions.append(str(pred))

		#draw the prediction on the output image
		cv2.rectangle(output, (x - 2, y - 2), 
			(x + w + 4, y + h + 4), (0, 255, 0), 1)
		cv2.putText(output, str(pred), (x - 5, y - 5), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

		#shows the output image
		print("[INFO] captcha: {}".format("".join(predictions)))
		cv2.imshow("Output", output)
		cv2.waitKey()