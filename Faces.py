import numpy as numpy
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('Classifiers/data/haarcascade_frontalface_alt2.xml')

#initializiing the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#bringing in trained data
recognizer.read("trainner.yml")
labels ={}

with open("labels.pickle",'rb') as f:
    #used this module to serialize adndeserialize any python objects.Converts the pythonnobject into a stream of characters before the actuall writing to files happen.
    og_labels=pickle.load(f)#label dictionary
    #inverting the key value pairs
    labels = {v:k for k,v in og_labels.items()}
    
cap=cv2.VideoCapture(0)

while(True):
    #frame by frame capturing
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    #interating through faces in the frames
    for (x,y,w,h) in faces:
        print(x,y,w,h) #making the region of interest
        roi_gray = gray[y:y+h, x:x+w] #staring from Y coordinate ot y+h coordinate
        roi_color = frame[y:y+h, x:x+w]
        
         # recognising the imge  using deep learnt model
        id_,conf = recognizer.predict(roi_gray)# predicting the roi
        
        if conf >=45 and conf<=85:
            print(id_)
            print(labels[id_])
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (255,0,0) #BGR 
        stroke = 3 # the thickness of the line
        end_code_x = x + w 
        end_code_y = y + h
        cv2.rectangle(frame, (x,y), (end_code_x,end_code_y),color,stroke)
        
       
    #display the resulting frame
    cv2.imshow('frame',frame)  #syntax:cv2.imshow(window_name, image) ,window_name: A string representing the name of the window in which image to be displayed.
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    
#to release the capture when everything is done
cap.release()
cv2.destroyAllWindows()