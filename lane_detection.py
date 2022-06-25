#!/usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneDetection():
    def __init__(self):
        rospy.init_node("lane_detect")
        rospy.Subscriber("/front_camera/color/image_raw",Image,self.cam_callback)
        self.bridge=CvBridge()
        rospy.spin()
        
    def cam_callback(self,msg):
        self.img=self.bridge.imgmsg_to_cv2(msg,"bgr8")
           
        #Convert to grayscale
        self.gray_img=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        
        #Apply basic thresolding.
        _,self.basic_thresh=cv.threshold(self.gray_img,190,255,cv.THRESH_BINARY)
        
        #Apply Gaussian Blur.
        self.blurred = cv.GaussianBlur(self.basic_thresh,(5,5),0.8)
        
        #Find line edges with C.E.D method.
        self.canny_img = cv.Canny(self.blurred,50,150)
        
#        ======================= Technique-1 : Create one ROI. ================
        #Create ROI.
#        mask = np.zeros_like(self.canny_img)
#        vertices = np.array([[(110,454),(260,284),(420,285),(556,413)]],np.int32)
#        cv.fillPoly(mask, vertices, 255)
#        self.roi_img = cv.bitwise_and(self.canny_img, mask)
#        
#        #Detect lines.
#        lines = cv.HoughLinesP(self.roi_img,2,np.pi/180,20,np.array([]),minLineLength=50,maxLineGap=200)
#        self.zeros = np.zeros_like(self.img)
#        for line in lines:
#            for x1,y1,x2,y2 in line:
#                cv.line(self.zeros,(x1,y1),(x2,y2),(0,0,255),2)
#        self.hough_img = cv.addWeighted(self.img,0.8,self.zeros, 1.0,0.)
#           
#        #Detect offset
#        h,w,d=self.img.shape
#        self.zeros=cv.cvtColor(self.zeros,cv.COLOR_BGR2GRAY)
#        _,self.zeros=cv.threshold(self.zeros,10,255,cv.THRESH_BINARY)
#        M=cv.moments(self.zeros)
#        cx=int(M['m10']/M['m00'])
#        cy=int(M['m01']/M['m00'])
#        cv.circle(self.hough_img,(cx,cy),5,(255,0,0),1)
#        cv.circle(self.hough_img,(int(w/2),int(h-10)),5,(0,0,255),-1)
#
#        cv.imshow("front_cam",self.hough_img)
#        cv.waitKey(1)
        #======================================================================

        #================ Technique-2 : Split lane lines. =====================
        #Detect left lane line.
        mask = np.zeros_like(self.canny_img)
        vertices = np.array([[(120,440),(260,275),(335,275),(335,440)]],np.int32)
        cv.fillPoly(mask, vertices, 255)
        self.roi_img = cv.bitwise_and(self.canny_img, mask)
        
        lines = cv.HoughLinesP(self.roi_img,2,np.pi/180,20,np.array([]),minLineLength=50,maxLineGap=200)
        self.zeros = np.zeros_like(self.img)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(self.zeros,(x1,y1),(x2,y2),(0,0,255),2)
                
        kernel=np.ones((15,15),'uint8')
        self.zeros=cv.dilate(self.zeros,kernel,iterations=1)
        kernel=np.ones((7,7),'uint8') 
        self.zeros=cv.erode(self.zeros,kernel,iterations=1)     
        
        self.left_lane = cv.addWeighted(self.img,0.8,self.zeros, 1.0,0.)
        #Calculate left lane line center.
        h,w,d=self.img.shape
        self.zeros=cv.cvtColor(self.zeros,cv.COLOR_BGR2GRAY)
        _,self.zeros=cv.threshold(self.zeros,10,255,cv.THRESH_BINARY)
        M=cv.moments(self.zeros)
        cx_left=int(M['m10']/M['m00'])
        cy_left=int(M['m01']/M['m00'])
        cv.circle(self.left_lane,(cx_left,cy_left),5,(255,0,0),1)
        cv.circle(self.left_lane,(int(w/2),int(h-10)),5,(0,0,255),-1)
        
        cv.imshow("left_lane",self.zeros)
        
        #Detect right lane line.
        mask = np.zeros_like(self.canny_img)
        vertices = np.array([[(335,440),(335,275),(415,275),(580,440)]],np.int32)
        cv.fillPoly(mask, vertices, 255)
        self.roi_img = cv.bitwise_and(self.canny_img, mask)
        
        lines = cv.HoughLinesP(self.roi_img,2,np.pi/180,20,np.array([]),minLineLength=50,maxLineGap=200)
        self.zeros = np.zeros_like(self.img)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(self.zeros,(x1,y1),(x2,y2),(0,0,255),2)
                
        kernel=np.ones((15,15),'uint8')
        self.zeros=cv.dilate(self.zeros,kernel,iterations=1)
        kernel=np.ones((7,7),'uint8') 
        self.zeros=cv.erode(self.zeros,kernel,iterations=1)
        
        self.right_lane = cv.addWeighted(self.img,0.8,self.zeros, 1.0,0.)
        #Calculate right lane line center.
        h,w,d=self.img.shape
        self.zeros=cv.cvtColor(self.zeros,cv.COLOR_BGR2GRAY)
        _,self.zeros=cv.threshold(self.zeros,10,255,cv.THRESH_BINARY)
        M=cv.moments(self.zeros)
        cx_right=int(M['m10']/M['m00'])
        cy_right=int(M['m01']/M['m00'])
        cv.circle(self.right_lane,(cx_right,cy_right),5,(255,0,0),1)
        cv.circle(self.right_lane,(int(w/2),int(h-10)),5,(0,0,255),-1)
        
        base_x=int((cx_left+cx_right)/2)
        base_y=int((cy_left+cy_right)/2)
        cv.circle(self.left_lane,(base_x,base_y),5,(255,0,0),1)
        cv.circle(self.right_lane,(base_x,base_y),5,(255,0,0),1)
        
        cv.imshow("right_lane",self.zeros)
        
        self.base_im=cv.addWeighted(self.left_lane,0.5,self.right_lane,0.5,0)
        
        cv.imshow("front_cam",self.base_im)
        cv.waitKey(1)
        #======================================================================
        
LaneDetection()