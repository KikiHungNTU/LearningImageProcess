# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:28:14 2018

@author: Ouch
"""

import numpy as np
import cv2

#from libraryCH.device.lcd import ILI9341

#lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=90)

cap = cv2.VideoCapture(0)
history = 20 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=history)
fgbg.setHistory(history)

while True:
        res, frame = cap.read()
        if not res:
            break

        fg_mask = fgbg.apply(frame)
        
        if frame < history:
            frame += 1
            continue
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 获取所有检测框
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if 500 < area < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)
        cv2.waitKey(1)

#while True:
#
#    ret, frame = cap.read()
#    fgmask = fgbg.apply(frame)
#    cv2.imshow('frame',fgmask)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
cap.release()
cv2.destroyAllWindows()


