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
plt.title("Original Image")
plt.imshow(wp.data,plt.cm.gray) # plot original image
plt.show()

limith = 25
limitv = 20
limitd = 20

zh=wp['h'].data
zh[zh<limith]=0.0
plt.title("Horizontal")
plt.imshow(zh,plt.cm.gray) # plot horizontal decomposition of image
plt.show()


zv=wp['v'].data
zv[zv<limitv]=0.0
plt.title("Vertical")
plt.imshow(zv,plt.cm.gray) # plot vertical decomposition of image
plt.show()

zd=wp['d'].data
zd[zd<limitd]=0.0
plt.title("Diagonal")
plt.imshow(zd,plt.cm.gray) # plot diagonal decmposition of image
plt.show()


zf = cv2.bitwise_or(zh, zd)
zf = cv2.bitwise_or(zf, zv)
cv2.imshow('Bitwise OR', zf)
#
## De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()


#zf[zf<0.0]=0.0
#plt.imshow(zf,plt.cm.gray) # plot final decmposition of image
#plt.show()