import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)

def readim(path):
  img = cv2.imread(path,0)

def fft():
f = np.fft.fft2(img) # fft to convert the image to freq domain
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)

x = 30 # 30 gives okayish results

# High Pass Filter (HPF)
fshift[crow-x:crow+x, ccol-x:ccol+x] = 0

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift) # inverse fft to get the image back 
img_back = np.abs(img_back)

# band pass part
y = 30
fshift2 = np.copy(fshift)
fshift2[0:y, 0:cols] = 0
fshift2[0:rows,0:y] = 0
fshift2[rows-y:rows,0:cols] = 0
fshift2[0:rows,cols-y:cols] = 0
f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = np.fft.ifft2(f_ishift2) # inverse fft to get the image back 
img_back2 = np.abs(img_back2)
# end band pass part

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# band pass output next
plt.subplot(133),plt.imshow(img_back2, cmap = 'gray')
plt.title('Image after BPF'), plt.xticks([]), plt.yticks([])

cv2.imwrite('HPF.jpg',img_back)
cv2.imwrite('BPF.jpg',img_back2)

plt.show()