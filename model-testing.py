import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('data/low_trained_model.h5')

# Load an image for face detection
frame = cv2.imread("data/neutral_boy.jpg")

# Load a pre-trained face detection cascade classifier
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around detected faces
for x, y, w, h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected faces
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Select the first detected face (you can modify this part for multiple faces)
face_roi = None
for (ex, ey, ew, eh) in faces:
    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

# Display the selected face (commented out for this code)
# plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

# Resize and preprocess the face image for classification
final_image = cv2.resize(face_roi, (224, 224))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image / 255.0

# Make predictions using the loaded model
predictions = loaded_model.predict(final_image)
predicted_class = np.argmax(predictions)

# Print the predicted class
print("Predicted Class value:", predicted_class)
print("""0 == Angry          4 == Sad 
1 == Disgust        5 == Surprise 
2 == Fear           6 == Neutral 
3 == Happy""")

# Show the recognized image
plt.show()