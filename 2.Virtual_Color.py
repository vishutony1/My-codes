import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)


my_colors = [[161,155,84,179,255,255],
             [57,76,0,100,255,255]]

my_color_values = [[0,0,255],
                   [0,255,0]]

my_points = []

def find_color(frame, my_colors, my_color_values):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    count = 0
    new_points = []
    for color in my_colors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(hsv, lower, upper)
        # cv2.imshow(str(color[0]), mask)
        x,y = getcontours(mask)
        cv2.circle(frame_result, (x,y), 10,my_color_values[count],cv2.FILLED)
        if x != 0 and y != 0:
            new_points.append([x,y,count])
        count += 1
    return new_points

def getcontours(frame):
    contours,_ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            # cv2.drawContours(frame_result, contour, -1, (255,0,0), 2)
            approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
            x,y,w,h = cv2.boundingRect(approx)
    return x+w//2,y

def drawoncanvas(my_colors, my_color_values):
    for point in my_points:
        cv2.circle(frame_result, (point[0], point[1]), 10, my_color_values[point[2]], cv2.FILLED)



while True:
    ret, frame = cap.read()
    frame_result = frame.copy()
    new_points = find_color(frame, my_colors, my_color_values)
    if len(new_points) != 0:
        for newP in new_points:
            my_points.append(newP)
    if len(my_points) != 0:
        drawoncanvas(my_colors, my_color_values)
    cv2.imshow("feed", frame_result)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()