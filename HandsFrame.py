# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:11:15 2019

@author: Vaidik
"""

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    
    try: #an error comes if it does not find anything in window as it cannot find contour of max area therefore this try error statement
        ret, frame = cap.read()
        #print("Frame: ",frame.width)
        frame=cv2.flip(frame,1)
        #cv2.imshow("Flip", frame)
        kernel_left = np.ones((3,3),np.uint8)
        kernel_right = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi_left=frame[300:600, 0:300]
        roi_right=frame[0:200, 339:639]
    
        cv2.imshow("Roi Right", roi_right)
        cv2.imshow("Roi Left", roi_left)
        
        cv2.rectangle(frame,(0, 300),(300,700),(0,255,0),0)
        cv2.rectangle(frame,(339, 0), (639, 200), (0, 255, 0), 0)
        
        hsv_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2HSV)
        hsv_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2HSV)
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask_left = cv2.inRange(hsv_left, lower_skin, upper_skin)
        mask_right = cv2.inRange(hsv_right, lower_skin, upper_skin)
   
        
    #extrapolate the hand to fill dark spots within
        mask_left = cv2.dilate(mask_left,kernel_left,iterations = 4)
        mask_right = cv2.dilate(mask_right, kernel_right, iterations = 4)
        
    #blur the image
        mask_left = cv2.GaussianBlur(mask_left,(5,5),100) 
        mask_right = cv2.GaussianBlur(mask_right,(5,5),100)
        
        
    #find contours
        _,contours_left,hierarchy_left= cv2.findContours(mask_left,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        _,contours_right,hierarcht_right= cv2.findContours(mask_right,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
        cnt_left = max(contours_left, key = lambda x: cv2.contourArea(x))
        cnt_right = max(contours_right, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt_left,True)
        epsilon_right = 0.0005*cv2.arcLength(cnt_right, True)
        approx= cv2.approxPolyDP(cnt_left,epsilon,True)
        approx_right = cv2.approxPolyDP(cnt_right, epsilon_right, True)
       
    #make convex hull around hand
        hull = cv2.convexHull(cnt_left)
        hull_right = cv2.convexHull(cnt_right)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areahull_right = cv2.contourArea(hull_right)
        areacnt = cv2.contourArea(cnt_left)
        areacnt_right = cv2.contourArea(cnt_right)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
        arearatio_right = ((areahull_right-areacnt_right)/areacnt_right)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        hull_right = cv2.convexHull(approx_right, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        defects_right = cv2.convexityDefects(approx_right, hull_right)
        
    # l = no. of defects
        l=0
        r=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi_left, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi_left,start, end, [0,255,0], 2)
            
            
        l+=1
        
        #print corresponding gestures which are in their ranges for left Hand
        print("L = ",l)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif arearatio<17.5:
                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
         
              if arearatio<27:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else:
                    cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)



        #For Right hand        
        for i in range(defects_right.shape[0]):
            s,e,f,d = defects_right[i,0]
            start = tuple(approx_right[s][0])
            end = tuple(approx_right[e][0])
            far = tuple(approx_right[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle_right = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle_right <= 90 and d>30:
                r += 1
                cv2.circle(roi_right, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi_right,start, end, [0,255,0], 2)
            
            
        r+=1
                
        
        print("R = ",r)
        if r==1:
            if areacnt_right<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio_right<12:
                    cv2.putText(frame,'0',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif arearatio_right<17.5:
                    cv2.putText(frame,'Best of luck',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                   
                else:
                    cv2.putText(frame,'1',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif r==2:
            cv2.putText(frame,'2',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif r==3:
         
              if arearatio_right<27:
                    cv2.putText(frame,'3',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else:
                    cv2.putText(frame,'ok',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif r==4:
            cv2.putText(frame,'4',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif r==5:
            cv2.putText(frame,'5',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif r==6:
            cv2.putText(frame,'reposition',(600,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(610,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            

            
        #show the windows
        cv2.imshow('Left Mask',mask_left)
        cv2.imshow("Right Mask", mask_right)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
    