# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
image = cv2.imread('1.jpg')
preprocessed = img_to_array(image, data_format=None)
cv2.imwrite('output.jpg',preprocessed)