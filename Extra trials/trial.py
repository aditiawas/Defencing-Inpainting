import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)

image = cv2.imread('BPF.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
cv2.imwrite('gray_image.png',gray_image)
x=np.asarray(Image.open("gray_image.png").convert("L"))

shape = x.shape
max_lev = 3 # how many levels of decomposition to draw
label_levels = 3 # how many levels to explicitly label on the plots

for level in range(1, max_lev + 1):
  # compute the 2D DWT
  c = pywt.wavedec2(x, 'db2', mode='sym', level=level)
  # normalize each coefficient array independently for better visibility
  c[0] /= np.abs(c[0]).max()
  for detail_level in range(level):
    c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]

  plt.imshow(c[0], cmap=plt.cm.gray)
  plt.title('Coefficients\n({} level)'.format(level))
  plt.show()

  print(c[0].shape)
  wp = pywt.WaveletPacket2D(data=c[0], wavelet='db1', mode='sym')
  limith = 0
  limitv = 0
  limitd = 0
  limitf = 25

  zh=wp['h'].data
  zh[zh<limith]=0.0
  print(zh.shape)
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
  print(zf.shape)
  res = cv2.resize (zf, dsize= (418, 240), interpolation=cv2.INTER_CUBIC)
  plt.title("ORed Image")
  plt.imshow(res) # plot final decomposition of image
  plt.show()