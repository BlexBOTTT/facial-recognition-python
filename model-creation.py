import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, Model

# Define data directory and classes
data_directory = "train_short/"
Classes = ["0"]

# Define image size for resizing
img_size = 224

# Create training data from images
training_data = []

def create_training_data():
    for category in Classes:
        path = os.path.join(data_directory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# Shuffle the training data
import random

random.shuffle(training_data)

# Separate features (x) and labels (y)
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3)  # Convert to 4-dimension
x = x / 255.0  # Normalize the images

y = np.array(y)

# Load a pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2()

# Create a new model by adding custom layers on top of MobileNetV2
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

new_model = tf.keras.Model(inputs=base_input, outputs=final_output)

# Compile the model
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
new_model.fit(x, y, epochs=1)   # higher epochs = more iterations hence increased accuracy

# Save the model
new_model.save('low_trained_model.h5')  # .h5 says legacy, recommended to use .keras
