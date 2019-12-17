import numpy as np
import cv2
import time

def nothing(x):
    pass

def roi(image,vert):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vert, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def pointRectColl(point, rect):
    if(point[0] > rect[0][0] and point[0] < rect[1][0] and point[1] > rect[0][1] and point[1] < rect[1][1]):
        return True
    return False

### Basic determinant to calculate area
def approxDeterminant(approx):
    leftCal = approx[len(approx) - 1][0][0] * approx[0][0][1]
    rightCal = approx[len(approx) - 1][0][1] * approx[0][0][0]
    for i in range(len(approx) - 1):
        leftCal += approx[i][0][0] * approx[i + 1][0][1]
        rightCal += approx[i][0][1] * approx[i + 1][0][0]
    result = 0.5 * (abs(leftCal - rightCal))
    return result

### Calculating center of mass (com)
def approxCom(approx):
    sumX = 0
    sumY = 0
    for i in range(len(approx)):
        sumX += approx[i][0][0]
        sumY += approx[i][0][1]
    return int(sumX/len(approx)), int(sumY/len(approx))

### Calculating average of the array containing the newest ten com values
def com(array):
    sumX = 0
    sumY = 0
    for i in range(len(array)):
        sumX += array[i][0]
        sumY += array[i][1]
    return int(sumX/len(array)), int(sumY/len(array))

def contourFilteringStage(contoursInputFrame,contoursOutputFrame,last,comArray,debug=True):
    contours , h = cv2.findContours(contoursInputFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    comAvrg = (0,0)
    now = round(time.time())
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) > 4 and len(approx)<15:
            calculatedDeterminant = approxDeterminant(approx)
            approxCal = approxCom(approx)
            if(calculatedDeterminant>50):
                last = round(time.time())
                comArray.append(approxCal)
                comAvrg = com(comArray)
                if(debug):
                    cv2.drawContours(contoursOutputFrame, [cnt], 0, (25, 25, 255), 2)

                ### After ten values array erases the first one slides it's values one left and the newest value is put as tenth value
                if(len(comArray) > 10):
                    comArray.pop(0)
        
        
                ### If two points too far away clear all the array
                if(pow(comAvrg[0] - approxCal[0],2)+pow(comAvrg[1] - approxCal[1],2) > 500):
                    comArray.clear()
        ### If the obj hasn't been shown for 3 sec reset the array###
    if(now - last > 3):
        comAvrg = (0,0)
    return last,comArray,comAvrg

def colorFilterStage(frame,filters):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array(filters[3:])
    upper_green = np.array(filters[:3])


    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    comArray = []
    comArrayCal = (0,0)
    last = 0
    while(1):
        _,frame = cam.read()
        res = colorFilterStage(frame,[93,255,255,51,39,0])
        last,comArray,comArrayCal = contourFilteringStage(res,frame,last,comArray)
        cv2.imshow("res",res)
        cv2.imshow("img",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()