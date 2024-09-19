import pyttsx3
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import csv
# from PIL import ImageGrab
 
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    

    global i
    global name_row
    
    i=0
    col_names = ['Name', '', 'Date', '', 'Time in','','Time out']
    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    timeStamp = now.strftime('%H:%M:%S')
    time=now.strftime('%H')
    
    attendance = [name, '', str(date), '', str(timeStamp),'',]
    if 6<=int(time)<=8:
        
        exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
        if exists:
            with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
                myDataList = csv.reader(csvFile1)
                nameList = []
                for row in myDataList:
                    nameList.append(row[0]) 
                   
            with open("Attendance\Attendance_" + date + ".csv", 'a',newline="") as csvFile1:
                if name not in nameList:
                    writer = csv.writer(csvFile1)
                    writer.writerow(attendance)
                csvFile1.close()
        else:
            with open("Attendance\Attendance_" + date + ".csv", 'a+',newline="") as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(col_names)
                writer.writerow(attendance)
    else: 
        with open("Attendance\Attendance_" + date + ".csv", 'r',newline="") as csvFile1:
                myDataList = csv.reader(csvFile1)
                nameList = []
                name_row={}
                nameList.extend(myDataList)
                
        i=0
        for row in nameList:
                    
                    if name in row:
                        if len(row)==6:
                            row.append(str(timeStamp))
                            print(row)
                            print(i)
                            name_row = {i:row}
                            break        
                    i+=1
        with open("Attendance\Attendance_" + date + ".csv", 'w',newline="") as b:
            writer = csv.writer(b)
            for line, row in enumerate(nameList):
                 data = name_row.get(line, row)
                 writer.writerow(data)
            csvFile1.close()
   
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)#for video capture

engine= pyttsx3.init() #for text to speech
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate',195 )
 
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)

while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            now = datetime.now()
            time = now.strftime('%H')
            if 6<=int(time)<=8:
                cv2.putText(img,"GOOD MORNING",(100,455),cv2.FONT_HERSHEY_TRIPLEX,1.8,(255, 0, 0),2)
                engine.say("GOOD MORNING")
                engine.say(str(name))
                
                engine.runAndWait()
            else:
                cv2.putText(img,"Good Afternoon...Have a nice day",(0,455),cv2.FONT_HERSHEY_TRIPLEX,1.1,(255, 0, 0),2)
                
                
                
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
            
            '''if name in nameList:   ATTENDANCE MARKED
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(30,255,255),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(30,255,255),cv2.FILLED)
                cv2.putText(img,"Attendance Marked",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)'''
                
        else:
           y1,x2,y2,x1 = faceLoc
           y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
           cv2.putText(img,"Access denied",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
    
        
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
