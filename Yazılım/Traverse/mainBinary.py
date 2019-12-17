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

def approxDeterminant(approx):
    leftCal = approx[len(approx) - 1][0][0] * approx[0][0][1]
    rightCal = approx[len(approx) - 1][0][1] * approx[0][0][0]
    for i in range(len(approx) - 1):
        leftCal += approx[i][0][0] * approx[i + 1][0][1]
        rightCal += approx[i][0][1] * approx[i + 1][0][0]
    result = 0.5 * (abs(leftCal - rightCal))
    return result

def colorFilterStage(frame,filters):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array(filters[3:])
    upper_green = np.array(filters[:3])


    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res

def contourFilteringStage(contourInput,contourOutput,debug=True):
    contours, h  = cv2.findContours(contourInput, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 or len(approx) == 5 or len(approx) == 6:
            cal = approxDeterminant(approx)
            #print("rec")
            if(cal > 50):
                region = roi(contourOutput, [cnt])
                regions.append(region)
                if(debug):
                    cv2.imshow("ROI", cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                cv2.drawContours(contourOutput, [cnt], 0, (25, 25, 255), 2)

        elif len(approx) > 12:
            #print("circle")
            cv2.drawContours(contourOutput, [cnt], 0, (255, 25, 25), 3)
        #else:
            #print("not rec")
    return regions



if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    regions=[]
    while(1):
        _,frame = cam.read()
        res = colorFilterStage(frame,[93,255,255,51,39,0])
        regions = contourFilteringStage(res,frame)
        cv2.imshow("res",res)
        cv2.imshow("img",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()