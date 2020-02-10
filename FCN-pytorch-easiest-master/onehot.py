import numpy as np
import cv2



def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

'''
img_name = '001100_mask.png'
# 001100.jpg
imgB = cv2.imread('./001100_mask.png', 0)
#imgB = cv2.imread('last_msk/'+img_name, 0)
imgB = cv2.resize(imgB, (160, 160))
#imgB = imgB/255
imgB = imgB.astype('uint8')
imgB = onehot(imgB, 6)
print(imgB.shape)
imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
print(imgB)
'''