import cv2
import os
from matplotlib import pyplot as plt
from skimage import color
from skimage import io

os.chdir("E:\\Documents\\College Work\\7th Semester\\Final Year Project")

img = cv2.imread('original.jpeg', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('BPF.jpg', cmap = 'gray')

img2 = color.rgb2gray(img2)
plt.imshow(img2)
plt.show()

a=cv2.inpaint(img,img2,-3,cv2.INPAINT_NS)
plt.imshow(a)

plt.show()
