import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition

path = "images"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread("cl")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return  encodeList

def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readline()
        nameList = []
        print(myDataList)
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")


encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frameS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGRA2RGB)

    facesCurFrame = face_recognition.face_locations(frameS)
    encodeCurFrame = face_recognition.face_encodings(frameS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4 , y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()