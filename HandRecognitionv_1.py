# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 23:57:07 2019

@author: Vaidik
"""

import cv2
import matplotlib.pyplot as plt

hand = cv2.imread('HandPalm1.jpg', 0)

retu, threshold = cv2.threshold(hand, 70, 255, cv2.THRESH_BINARY)

img, contours, hera = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('Returened Image', img)
print("\nHierarchy: ", hera)
plt.imshow(hera)

hull = [cv2.convexHull(c) for c in contours]

final = cv2.drawContours(hand, hull, -1, (255, 0, 0))

cv2.imshow('Original Image', hand)
cv2.imshow('Thrsold Image', threshold)
cv2.imshow('Final Output', final)

cv2.waitKey(0)
cv2.destroyAllWindows()