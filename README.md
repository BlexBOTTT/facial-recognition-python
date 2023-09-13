
 # Basic Face Detection in python with OpenCV
 
<<<<<<< HEAD
 ### Description
- This Python script utilizes OpenCV to perform basic face detection function to open one computer's camera.
=======
 ## Description
- This Python script utilizes OpenCV to perform basic ~~face detection~~ function to open one computer's camera.
>>>>>>> 054e252db70b71f6abd98b7d1fd120cfaa3f61f4

### Feature
- Opens the camera
- Detects person's face, marked via rectangle-square shape marker

## Usage
  - When run, it opens a camera window displaying the live feed.
  - When a person's face is detected, there will be a white, thin square following the face
    ~~(just like with a typical modern smartphone camera)~~
- **Termination:**
  - To exit the program, press the 'q' key.

<<<<<<< HEAD
### Limitations:
- **No Capture and save image:**
    - It doesn't capture images from the live camera; it's purely for real-time face detection,
  let alone saving that captured picture.
- **Lighting and eye-wear:**
    - For some unknown reason, a low-lit face especially with eye-wear may not be recognized effectively,
    until the face subject is well lit or take off subject's eye-wear.
=======
## Downsides:
- **No Capture:**
  - It doesn't capture images; it's purely for real-time face detection.
>>>>>>> 054e252db70b71f6abd98b7d1fd120cfaa3f61f4
- **Dependencies:** 
     - Requires installation of OpenCV.
- **Pre-trained detection model:**
    - The pre-trained face detection model used in this project is provided by the [OpenCV project](https://github.com/opencv/opencv).
      - Model File: `data/haarcascade_frontalface_default.xml`


## ~~tools mentioned (can be used less)~~
- python

### Either of
- pycharm
- ~~jupyter~~

- ~~anaconda~~
- pip
- opencv
- ~~keras (for dataset)~~
- ~~tensorflow~~
