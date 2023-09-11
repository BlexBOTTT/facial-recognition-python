import cv2

cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection


    # Display the frame with rectangles
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()