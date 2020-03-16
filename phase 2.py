import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = cv2.imread('BPF.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to double
cv2.imwrite('gray_image.png',gray_image)
x=np.asarray(Image.open("gray_image.png").convert("L")) # get image in array format
wp = pywt.WaveletPacket2D(data=x, wavelet='db1', mode='sym') # define wavelet package
plt.imshow(wp.data,plt.cm.gray) # plot original image
plt.show()
z=wp['h'].data
z[z<0.0]=0.0
plt.imshow(z,plt.cm.gray) # plot horizontal decomposition of image
plt.show()
z=wp['v'].data
z[z<0.0]=0.0
plt.imshow(z,plt.cm.gray) # plot vertical decomposition of image
plt.show()
z=wp['d'].data
z[z<0.0]=0.0
plt.imshow(z,plt.cm.gray) # plot diagonal decmposition of image
plt.show()