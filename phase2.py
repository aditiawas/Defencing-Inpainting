import cv2
import pywt # PyWavelet library
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import os

def processmask(original):

  dir = os.path.dirname(__file__)
  foldername = os.path.join(dir, 'Images')
  os.chdir(foldername)

  x=np.asarray(Image.open('HPF.jpg').convert("L"))
  #xdim = int(x.shape[1]/500)
  #print(x.shape)
  #if xdim == 0:
  #  xdim = 1
  #change1 = int(x.shape[1]/xdim)
  #change2 = int(x.shape[0]/xdim)
  #if change1 % 2 == 1:
  #  change1 = change1 + 1
  #if change2 % 2 == 1:
  #  change2 = change2 + 1
  #x = cv2.resize(x, dsize=(change1, change2), interpolation=cv2.INTER_AREA)

  print(x.shape)
  wp = pywt.WaveletPacket2D(data=x, wavelet='db1', mode='sym')
  # plt.title("Mask from Phase 1")
  # plt.imshow(wp.data,plt.cm.gray) # plot original image
  # plt.show()

  limith = -10
  limitv = -10
  limitd = -10
  limitf = 15

  zh=wp['h'].data
  zh[zh<limith]=0.0
  zv=wp['v'].data
  zv[zv<limitv]=0.0
  zd=wp['d'].data
  zd[zd<limitd]=0.0

  plt.subplot(131),plt.imshow(zh,plt.cm.gray) # plot horizontal decomposition
  plt.title("Horizontal Component"), plt.xticks([]), plt.yticks([])
  plt.subplot(132),plt.imshow(zv,plt.cm.gray) # plot vertical decomposition
  plt.title("Vertical Component"), plt.xticks([]), plt.yticks([])
  plt.subplot(133),plt.imshow(zd,plt.cm.gray) # plot diagonal decomposition
  plt.title("Diagonal Component"), plt.xticks([]), plt.yticks([])
  plt.show()

  plt.gray()

  zf = zh + zv + zd
  zf[zf<limitf] = 0.0
  zf[zf>limitf] = 20
  # print(zf.shape)
  
  plt.title("Decompositions ORed")
  plt.imshow(zf) # plot final OR of decomposition of image
  plt.show()
  plt.imsave('basic mask.jpg',zf)

  origimg = np.asarray(Image.open(original))
  res = cv2.resize (origimg, dsize= (zf.shape[1], zf.shape[0]) , interpolation=cv2.INTER_AREA)
  plt.imsave('applyon.jpg',res)

  thickness = 2
  res = np.array(cv2.morphologyEx(zf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))))
  # print(res.size)
  plt.title("Final fence mask")
  plt.imshow(res)
  plt.show()

  plt.imsave("finalmask.jpg", res)