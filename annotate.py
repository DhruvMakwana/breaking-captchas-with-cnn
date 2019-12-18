#importing libraries
#to run this program run following command in command line
#python annotate.py --input downloads --annot dataset
from imutils import paths
import argparse
import imutils
import cv2
import os

#argument parser for command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="Path to input directory of images")
#--input is the path to our raw captcha images
ap.add_argument("-a", "--annot", required=True,
    help="path to output directory of annotations")
#--annot is path to where we’ll be storing the labeled digits
args = vars(ap.parse_args())

#grab the image paths then initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

#loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

	try:
		#converting image to grayscale
		image = cv2.imread(imagePaths)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
		#cv2.copyMakeBorder() method is used to create a border around 
		#the image like a photo frame.
		#cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
		#The row or column at the very edge of the original is replicated 
		#to the extra border.

		#threshold the image to reveal the digits
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV  | cv2.THRESH_OTSU)[1]
		#Thresholding is a technique in OpenCV, which is the assignment
		#of pixel values in relation to the threshold value provided. 
		#In thresholding, each pixel value is compared with the 
		#threshold value. If the pixel value is smaller than the 
		#threshold, it is set to 0, otherwise, it is set to a maximum 
		#value (generally 255).
		#cv2.thresholding(src, thresholdValue, maxVal, thresholdingTechnique)

		#find contours in the image, keeping only the four largest ones
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
		#Contours are defined as the line joining all the points along 
		#the boundary of an image that are having the same intensity. 
		
		#loop over the contours
		for c in cnts:
			#computing bounding box for the contour before extract the digit
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

			#display the character, making it larger enough for us
			#to see, then wait for a keypress
			cv2.imshow("ROI", imutils.resize(roi, width=28))
			key = cv2.waitKey(0)

			#boundingRect Calculates the up-right bounding rectangle 
			#of a point set.

			if key == ord("`"):
				print("[Info] ignoring character")
				continue

			#grab the key that was pressed and construct the path
			#the output directory
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])

			#if the output directory does not exist create it
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)

			#write the labeled character to file
			counts = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(
				str(count).zfill(6))])
			cv2.imwrite(p, roi)

			#increment the count for the current key
			counts[key] = counts + 1

			#we are trying to control-c out of the script, so break from the
			#loop (you still need to press a key for the active window to
			#trigger this)
	except KeyboardInterrupt:
		print("[Info] manually leaving script")
		break

	except:
		print("[Info] skipping image...")

