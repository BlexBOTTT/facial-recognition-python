# import opencv
import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize video capture source with index marking
cap = cv2.VideoCapture(0) # 0 == default camera

# Set the desired camera resolution
width, height = 1200, 720     # Adjust these values to your preferred resolution
cap.set(3, width)       # Set the width
cap.set(4, height)      # Set the height


while True:
    # Read a frame from the camera
    ret, frame = ca720p.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
    # (###, ###, ###) = change to various color via RGB values
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the frame with rectangles
    cv2.imshow('Face Detection, not recognition (press q to exit.)', frame)

    # code for window termination when key "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
