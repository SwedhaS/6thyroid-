# import the necessary packages
import cv2

image = cv2.imread('1.jpg')
preproceesed = cv2.resize(image, (128, 128),
	interpolation=cv2.INTER_AREA)
cv2.imwrite('output.jpg',preproceesed)