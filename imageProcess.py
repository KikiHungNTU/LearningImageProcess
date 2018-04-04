# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:29:08 2018

@author: Ouch
"""

import cv2

path = 'C:/Users/chich/Desktop/learningImage/'
pic = 'lena.png'

#灰階開圖
image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)

#視窗呈現圖
cv2.imshow('my image', image)
cv2.waitKey()
cv2.destroyAllWindows()

'''
用numpy進行灰階處理
'''
#先將圖片讀成矩陣
import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt

def convert_Gray(image):
    #打開已經是ndArray;unit8
    img = Image.open(image)
    #img.mode
    #img.getpixel((0,0))
    imgArray = np.copy(img)
    imgArray = imgArray.astype(np.float64, copy = False)
    imgSum = np.copy(imgArray)
    
    #Compute GrayImage Y
    imgSum[:,:,0] *= 0.299
    imgSum[:,:,1] *= 0.587
    imgSum[:,:,2] *= 0.114
    
    imgSum = imgSum.sum(2)
    img_new = Image.fromarray(imgSum.astype(np.uint8))
    img_new.show()
    return img_new
    


        
    