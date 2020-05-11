import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

dir = os.path.dirname(__file__)
foldername = os.path.join(dir, 'Images')
os.chdir(foldername)
image = np.array(cv2.imread('basic mask.jpg'))

res = np.array(cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))))

print(res.size)

plt.imshow(res)
plt.show()

plt.imsave("finalmask.jpg", res)