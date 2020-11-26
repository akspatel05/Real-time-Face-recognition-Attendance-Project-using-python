import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime
path = 'KNOWN_FACES/MUSK'
known_faces = []
known_names = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    known_faces.append(curImg)
    known_names.append(os.path.splitext(cl)[0])
print(known_names)

def findEncodings(known_faces):
    encodeList = []
    for img in known_faces:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        if len(encodeList) > 0:
            biden_encoding = encodeList[0]
        else:
            print("No faces found in the image!")
            quit()
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(known_faces)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    faceCurframe = face_recognition.face_locations(imgS)
    encodeCurframe = face_recognition.face_encodings(imgS, faceCurframe)

    for encodeFace,faceLoc in zip(encodeCurframe,faceCurframe):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
       # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = known_names[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
