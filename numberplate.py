import numpy as np
import cv2
from PIL import Image
import pytesseract as tess

img=cv2.imread('images/test.jpg')
img2 = cv2.GaussianBlur(img, (3,3), 0)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)	
_,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("input",img)
cv2.imshow("input2",img2)

cv2.waitKey(0)