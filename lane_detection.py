#!/usr/bin/env python3

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32

class LaneDetection():
    def __init__(self):
        rospy.init_node("lane_detect")
        rospy.Subscriber("/front_camera/color/image_raw",Image,self.cam_callback)
        self.pub=rospy.Publisher("/lane_offset",Int32,queue_size=1)
        self.bridge=CvBridge()
        rospy.spin()
        
    def cam_callback(self,msg):
        self.img=self.bridge.imgmsg_to_cv2(msg,"bgr8")       
        self.h,self.w,self.d=self.img.shape
        self.gray_img=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        _,self.basic_thresh=cv.threshold(self.gray_img,190,255,cv.THRESH_BINARY)
        self.blurred = cv.GaussianBlur(self.basic_thresh,(5,5),0.8)
        self.canny_img = cv.Canny(self.blurred,50,150)
        self.process_left_lane_line()
        self.process_right_lane_line()
        self.calc_lane_center()
    
    def erode_dilate_proc(self,img):
        kernel=np.ones((15,15),'uint8')
        img=cv.dilate(img,kernel,iterations=1)
        kernel=np.ones((7,7),'uint8') 
        img=cv.erode(img,kernel,iterations=1)
        return img
    
    def hough_line(self,img):
        lines = cv.HoughLinesP(img,2,np.pi/180,20,np.array([]),minLineLength=50,maxLineGap=200)
        zeros = np.zeros_like(self.img)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv.line(zeros,(x1,y1),(x2,y2),(0,0,255),2)
        return zeros
     
    def calc_moment(self,img):
        tmp=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _,thresh=cv.threshold(tmp,10,255,cv.THRESH_BINARY)
        M=cv.moments(thresh)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        return cx,cy
    
    def create_roi(self,arr,img):
        mask = np.zeros_like(img)
        vertices = np.array(arr,np.int32)
        cv.fillPoly(mask,vertices, 255)
        roi_img = cv.bitwise_and(self.canny_img,mask)
        return roi_img
    
    def process_left_lane_line(self):
        roi_img=self.create_roi([[(120,440),(260,275),(335,275),(335,440)]], self.canny_img)         
        lines=self.hough_line(roi_img)
        morphed=self.erode_dilate_proc(lines)
        self.left_lane = cv.addWeighted(self.img,0.8,morphed, 1.0,0.)      
        self.cx_left,self.cy_left=self.calc_moment(morphed)      
        cv.circle(self.left_lane,(self.cx_left,self.cy_left),5,(255,0,0),1)
        cv.circle(self.left_lane,(int(self.w/2),int(self.h-10)),5,(0,0,255),-1)      
#        cv.imshow("left_lane",zeros)
   
    def process_right_lane_line(self): 
        roi_img=self.create_roi([[(335,440),(335,275),(415,275),(580,440)]], self.canny_img)
        lines=self.hough_line(roi_img)              
        morphed=self.erode_dilate_proc(lines)      
        self.right_lane = cv.addWeighted(self.img,0.8,morphed, 1.0,0.)
        self.cx_right,self.cy_right=self.calc_moment(morphed)      
        cv.circle(self.right_lane,(self.cx_right,self.cy_right),5,(255,0,0),1)
        cv.circle(self.right_lane,(int(self.w/2),int(self.h-10)),5,(0,0,255),-1)      
#        cv.imshow("right_lane",self.zeros)
        
    def calc_lane_center(self):
        base_x=int((self.cx_left+self.cx_right)/2)
        base_y=int((self.cy_left+self.cy_right)/2)      
        cv.circle(self.left_lane,(base_x,base_y),5,(255,0,0),1)
        cv.circle(self.right_lane,(base_x,base_y),5,(255,0,0),1)               
        self.base_im=cv.addWeighted(self.left_lane,0.5,self.right_lane,0.5,0)               
        offset=((self.cx_left+self.cx_right)/2)-int(self.w/2)       
#        print("Offset : ",offset)
        self.pub.publish(int(offset))        
        cv.imshow("front_cam",self.base_im)
        cv.waitKey(1)
        
LaneDetection()