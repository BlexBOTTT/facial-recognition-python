#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf 
import cv2 # pip install opencv-python
#pip install opencv-contrib-python full package
#from deepface import DeepFace #pip install deepface
import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
#set the rectangle background to white
rectangle_bgr = (255, 255, 255)
#make a black image
img = np.zeros((500, 500))
#set some text
text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
#make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

    
# Load the pre-trained model for emotion recognition
new_model = tf.keras.models.load_model('Final_model_95p07.h5')  # Replace 'your_model_path.h5' with the actual path to your model file


while True:
    ret, frame = cap.read()
    #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    face_roi = None
    final_image = None
    
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
                
    if face_roi is not None:
        final_image = cv2.resize(face_roi, (224, 224)) 
        final_image = np.expand_dims(final_image, axis=0) ## need fourth dimension
        final_image = final_image/255.0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if face_roi is not None:
        Predictions = new_model.predict(final_image)
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN
    
        if(np.argmax(Predictions)==0):
            status = "Angry"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(Predictions)==1):
            status = "Disgust"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(Predictions)==2):
            status = "Fear"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(Predictions)==3):
            status = "Happy"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(Predictions)==4):
            status = "Sad"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(Predictions)==5):
            status = "Surprise"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        else:
            status = "Neutral"
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Addd text
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
            cv2.imshow('Face Emotion Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




