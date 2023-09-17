import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model for emotion recognition
new_model = tf.keras.models.load_model('data/low_trained_model.h5')

# Load pre-trained face detection model (Haar Cascade classifier)
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize the camera (video capture)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set the desired camera resolution (optional)
# width, height = 1200, 720  # Adjust these values to your preferred resolution
# cap.set(3, width)  # Set the width
# cap.set(4, height)  # Set the height

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        # Extract the region of interest (ROI) for the detected face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Perform emotion recognition for the detected face
        face_roi = roi_color
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = new_model.predict(final_image)

        # Configure font for displaying emotions
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN

        # Define emotion labels
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Get the emotion with the highest prediction score
        status = emotions[np.argmax(Predictions)]

        # Display the detected emotion as text
        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

    # Display the frame with face detection and emotion recognition
    cv2.imshow('Face Emotion Recognition', frame)

    # Check if the 'q' key is pressed to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
