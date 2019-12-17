import cv2
import numpy as np

def filteringStage(frame):
    return cv2.cvtColor(cv2.medianBlur(frame,11),cv2.COLOR_BGR2GRAY)

def getCircles(frame,params=[40,50,0,0]):
    return np.uint16(cv2.HoughCircles(frame,cv2.HOUGH_GRADIENT,1,20,param1=params[0],param2=params[1],minRadius=params[2],maxRadius=params[3]))

def distCalculating(circle):
    return (170/circle[2])*30

def drawCircles(frame,circle):
    cv2.circle(frame,(circle[0],circle[1]),circle[2],(0,255,0),2)

