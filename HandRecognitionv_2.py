# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 00:22:01 2019

@author: Vaidik
"""

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while cap.isOpened():
    rect, frame = cap.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 1)
    crop_img = frame[100:300, 100:300]
    
    #Apply blur for image smoothing
    blur = cv2.GaussianBlur(crop_img, (3, 3), 0)
    
    #Change in HSV for better color extraction
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    #Create mask for defining lower & upper value for hand color of blur window
    mask = cv2.inRange(hsv_img, np.array([2, 0, 0]), np.array([20, 255, 255]))
    
    #defining kernel for avoid loss of pixel in image (smooth edges)
    kernel = np.ones((5, 5))
    
    #Apply Morfological function - Dialation(Adding of worthy pixels) & Erosion(Removing unworthy pixels)
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    
    #Appling Gaussian Blur for smoothing image if any extra pixels add during Morfological function
    filterd_img = cv2.GaussianBlur(erosion, (3, 3), 0)
    
    #Appling Thresolding (set a value for recognize the hand)
    rect, thresold = cv2.threshold(filterd_img, 127, 255, 0)
    cv2.imshow('Thresold img', thresold)
    
    #Find Contours(border of the hand)
    image, contours, heirarchy = cv2.findContours(thresold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Start Try(because it's good programing habbits :))
    try:
        #Finding contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        
        #Creating boudning rectangle around contour(border)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_img, (x, y), (x+w, y+h), (255, 0, 0), 0)
        
        #Finding convex hull
        hull = cv2.convexHull(contour)
        
        #Draw Contour on image
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
        
        #Defining defect(anglepoint between two fingers)
        hull = cv2.convexHull(contour, returnPoints = False)
        defects = cv2.convexityDefects(contour, hull)
        
        #Calculate defects(Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the fingertips) for all defects)
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0] #starting, endind, farest point & distance
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14 #formula of cosine [theta = cos(((b^2)+(c^2) - (a^2))/2bc) * (180/3.14)] & convert it in degree
            
            # if angle > 90 draw a circle at the far point 
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 3, [0, 0, 255], -1)

            cv2.line(crop_img, start, end, [0, 255, 0], 2)


        print("Number Of Defects: ", count_defects)
        # Print number of fingers

        if count_defects == 0:

            cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)

        elif count_defects == 1:

            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 2:

            cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 3:

            cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 4:

            cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        else:
            print("In Else")
            pass

    except:
        print("In Except")
        pass



    # Show required images

    cv2.imshow("Gesture", frame)

    all_image = np.hstack((drawing, crop_img))

    cv2.imshow('Contours', all_image)



    # Close the camera if 'q' is pressed

    if cv2.waitKey(1) == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()