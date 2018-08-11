# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 21:07:57 2018

@author: YQ
"""

import cv2
import numpy as np
import imutils
from keras.models import load_model
from imutils import contours

def preprocess(img):
    if img.shape[0] > img.shape[1]:
        img = imutils.resize(img, height=20)
    else:
        img = imutils.resize(img, width=20)
        
    padW = int((28 - img.shape[1]) / 2.0)
    padH = int((28 - img.shape[0]) / 2.0)
    
    image = cv2.copyMakeBorder(img, padH, padH, padW, padW, cv2.BORDER_CONSTANT, value=0)
    image = cv2.resize(image, (28, 28))
    
    return image
    

model = load_model("save/lenet_BS64_epoch3.h5")   # load the model

img = cv2.imread("test.png")  # load image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cnts = contours.sort_contours(cnts)[0]

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    
    out = thresh[y:y+h, x:x+w]
    out = preprocess(out)
    
    show = out.copy()
    
    out = np.expand_dims(out, 0)  # add first (batch) dimension
    out = np.expand_dims(out, -1)  # add last (channels) dimension
    
    sm = model.predict(out)
    prediction = np.argmax(sm, axis=-1)
    print(prediction)
      
    cv2.imshow("Out", show)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()