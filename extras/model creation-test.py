#!/usr/bin/env python
# coding: utf-8

# In[86]:


import tensorflow as tf 
import cv2 
import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# In[63]:


img_array = cv2.imread("train/0/Training_3908.jpg")


# In[64]:


img_array.shape ## rgb

#print(img_array)


# In[65]:


plt.imshow(img_array)


# In[66]:


data_directory = "train_short/"
# Classes = ["0", "1", "2", "3", "4", "5", "6"]
Classes = ["0", "1", "2", "3", "4", "5", "6"]

for category in Classes:
    path = os.path.join(data_directory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        #backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[67]:


img_size = 224 
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()


# In[68]:


new_array.shape


# # READ ALL IMAGES -> CONVERT TO ARRAY

# In[69]:


training_Data = []

def create_training_Data():
  for category in Classes:
    path = os.path.join(data_directory, category)
    class_num = Classes.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (img_size, img_size))
        training_Data.append([new_array, class_num])
      except Exception as e:
        pass


# In[70]:


create_training_Data()


# In[71]:


print(len(training_Data))


# In[71]:


# Convert to a NumPy array
# temp = np.array(training_Data)

# Check the shape of the resulting NumPy array
# print(temp.shape)


# In[40]:


import random

random.shuffle(training_Data)


# In[41]:


x = []
y = []

for features, label in training_Data:
        x.append(features)
        y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3) # convert to 4-dimesion

x.shape


# In[42]:


x = x/255.0


# In[43]:


type(y)

y = np.array(y)

y.shape


# # deep learning!

# In[111]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2


# In[112]:


model = tf.keras.applications.MobileNetV2()

model.summary()


# # TRANSFER LEARNING!

# In[113]:


base_input = model.layers[0].input
base_output = model.layers[-2].output

base_output


# In[47]:


final_output = layers.Dense(128)(base_output)      # adding a new layer, after the output of the global pooling layer
final_ouput = layers.Activation('relu')(final_output)      # activation function
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_ouput)  # 7 classes

final_output


# In[48]:


new_model = tf.keras.Model(inputs=base_input, outputs=final_output)


# In[49]:


new_model.summary()


# In[50]:


new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[51]:


new_model.fit(x, y, epochs = 10)


# In[52]:


new_model.save('Final_model_95p07.h5')


# In[53]:


new_model.save('Final_model_95p07.keras')


# In[55]:


new_model.save('Final_model_95p07.h5')


# In[114]:


loaded_model = tf.keras.models.load_model('Final_model_95p07.h5')


# In[140]:


frame = cv2.imread("../data/surprise_man.jpg")


# In[141]:


frame.shape


# In[142]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# # FACE DETECTION ALGORITHM

# In[123]:


# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[143]:


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[144]:


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[146]:


faces = faceCascade.detectMultiScale(gray,1.1,4)

for x, y, w, h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2) # BGR = 255, 255, 255 == white
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
            
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[147]:


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


# In[148]:


final_image = cv2.resize(face_roi, (224,224)) ##
final_image = np.expand_dims (final_image, axis =0) ## need fourth dimension 
final_image = final_image/255.0 ## normalizing


# In[150]:


Predictions = new_model.predict(final_image)


# In[151]:


Predictions[0]


# In[152]:


np.argmax(Predictions)


# In[ ]:




