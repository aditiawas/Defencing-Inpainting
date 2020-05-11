import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.transform import resize

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)

image = cv2.imread('BPF.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)
x=np.asarray(Image.open("gray_image.png").convert("L"))

print(x.shape)
wp = pywt.WaveletPacket2D(data=x, wavelet='db1', mode='sym')
plt.title("Original Image")
plt.imshow(wp.data,plt.cm.gray) # plot original image
plt.show()

limith = -10
limitv = -10
limitd = -10
limitf = 15

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
plt.imshow(zd,plt.cm.gray) # plot diagonal decomposition of image
plt.show()

plt.gray()
zf = zh + zv + zd
zf[zf<limitf] = 0.0
zf[zf>limitf] = 20
print(zf.shape)
plt.title("Final Image")
plt.imshow(zf) # plot final decomposition of image
print(zf.shape)
plt.show()
plt.imsave('basic mask.jpg',zf)

original=np.asarray(Image.open("original.jpeg"))
res = cv2.resize (original, dsize= (zf.shape[1], zf.shape[0]) , interpolation=cv2.INTER_AREA)

plt.imshow(original)
plt.show()
plt.imshow(res)
plt.show()
plt.imsave('apply on.jpg',res)

res = np.array(cv2.morphologyEx(zf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (12,12))))
print(res.size)
plt.imshow(res)
plt.show()

plt.imsave("finalmask.jpg", res)