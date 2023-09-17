
# Basic Face Recognition in Python (with extensions1)
 
# CREDITS:
 - For the [main baseline](https://www.youtube.com/watch?v=avv9GQ3b6Qg) of the work: 
 - [External source](https://medium.com/analytics-vidhya/realtime-face-emotion-recognition-using-transfer-learning-in-tensorflow-3add4f4f3ff3) but with similar code written:
 - OpenAI's [ChatGPT](https://openai.com/chatgpt) for big-time assistance in debugs, and possible optimization of the code.

## Description
- This Python script utilizes models to perform basic face recognition function from  one computer's camera.

## Feature
- Opens the camera
- Detects person's face, marked via rectangle-square shape marker and also indicated by the face written in the subject's face.

## Usage
  - When run, it opens a camera window displaying the live feed.
  - When a person's face is detected, there will be a thin square following the face
    ~~(just like with a typical modern smartphone camera)~~ and also an indication on whether which emotion the face convey.
- **Termination:**
  - To exit the program, press the 'q' key.

## Limitations:
- **No Capture and save image:**
    - It doesn't capture images from the live camera; it's purely for real-time face recognition ~~detection~~,
  let alone saving that captured picture.
- **Lighting and eye-wear:**
    - For some unknown reason, a low-lit face especially with eye-wear may not be recognized effectively,
    until the face subject is well lit or take off subject's eye-wear.
- **Dependencies:** 
     - Requires installation of OpenCV, Keras, Tensorflow, matplotlib, and other imports like os and numpy to properly work.
- **Pre-trained detection model:**
    - The pre-trained face detection model used in this project is provided by the [OpenCV project](https://github.com/opencv/opencv).
      - Model File: `data/haarcascade_frontalface_default.xml`
    - THE .H5/KERAS MODEL USED IN THIS REPOSITORY IS VERY LOW-TRAINED, MEANING IT HAS LESS DATASET TO FEED DUE TO THE HARDWARE AND TIME CONSTRAINTS.


