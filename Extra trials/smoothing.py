import cv2
import os

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)
image = cv2.imread('basic mask.jpg')

blur = cv2.medianBlur(image, 1)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,125, 255,cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.imshow('close', close)
cv2.imwrite('result.png', close)
cv2.waitKey()