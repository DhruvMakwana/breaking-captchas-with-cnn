#importing libraries
import imutils
import cv2

def preprocess(image, width, height):
	#grab the dimensions of the image, then initialize
	#the padding values
	(h, w) = image.shape[:2]

	#if the width is greater than height then resize along the width
	if w > h:
		image = imutils.resize(image, width=width)
	else:
		image = imutils.resize(image, height=height)	

	#determining padding values for width and height to 
	#obtain the target dimensions
	padW = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	#pad the image then apply one more resizing to handle any
	#rounding issues
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW, 
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))
	#cv2.copyMakeBorder() method is used to create a border around 
	#the image like a photo frame.
	#cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
	#The row or column at the very edge of the original is replicated 
	#to the extra border.

	return image
