import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"Images")

face_cascade = cv2.CascadeClassifier('Classifiers/data/haarcascade_frontalface_alt2.xml')

#training the recogniser 
recognizer = cv2.face.LBPHFaceRecognizer_create()





current_id = 0

label_ids ={
    
}
y_labels = []
x_train = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            #label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            label = os.path.basename(root).replace(" ", "-").lower()
            print(path)
            print("===================")
            print(label,path)
            
            if  not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
        
            id_= label_ids[label]
            print(label_ids)
            
            pil_image = Image.open(path).convert('L')#grayScale
            size =(550,550)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            
            image_array = np.array(pil_image,"uint8") #every pixel value is turned into a number stored in a numpy array(turn the grayscale into numpy array)
            print(image_array)
            #to detect the face in the image
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
print(y_labels)
                
                
with open("labels.pickle",'wb') as f:
    #used this module to serialize adndeserialize any python objects.Converts the pythonnobject into a stream of characters before the actuall writing to files happen.
    pickle.dump(label_ids,f)
    

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")