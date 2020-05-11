import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)

image = cv2.imread('BPF.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)
x=np.asarray(Image.open("gray_image.png").convert("L"))
wp = pywt.WaveletPacket2D(data=x, wavelet='db1', mode='sym')
plt.imshow(wp.data,plt.cm.gray) # plot original image
plt.show()
h=wp['h'].data
h[h<0.0]=0.0
plt.imshow(h,plt.cm.gray) # plot horizontal decomposition of image
plt.show()

#h1 = np.asarray(Image.fromarray(h).convert("L"))
#plt.imshow(h1) # plot horizontal decomposition of image
#plt.show()
#(thresh, im_bw) = cv2.threshold(h1, 128, 255, cv2.THRESH_BINARY | cv2.#THRESH_OTSU)
#cv2.imwrite('binary_image.png', im_bw)


v=wp['v'].data
v[v<0.0]=0.0
plt.imshow(v,plt.cm.gray) # plot vertical decomposition of image
plt.show()
d=wp['d'].data
d[d<0.0]=0.0
plt.imshow(d,plt.cm.gray) # plot diagonal decmposition of image
plt.show()


im_or = cv2.bitwise_or(h, d)
im_or = cv2.bitwise_or(im_or, v)
plt.title("ORed image")
plt.imshow(im_or,plt.cm.gray)
plt.show()