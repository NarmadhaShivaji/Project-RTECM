import cv2
#import numpy as np
import pyautogui
# Load the pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
# Accessing camera
cam = cv2.VideoCapture(0)
# Get the width and height of the monitor screen
screen_w, screen_h = pyautogui.size()
# Read every frame of the video at runtime
while True:
    # Read the data from the camera
    ret, frame = cam.read()
    if not ret:
        break
    # Convert frame color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Iterate through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255, 0), 2)
            # Calculate the center of the eye
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            # Normalize eye coordinates to screen size
            screen_x = eye_center_x * screen_w // frame.shape[1]
            screen_y = eye_center_y * screen_h // frame.shape[0]
            # Move the mouse based on eye position
            pyautogui.moveTo(screen_x, screen_y)
    # Display the frame
    cv2.imshow('Eye Controlled Mouse', frame)
    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the camera and close all OpenCV windows``
cam.release()
cv2.destroyAllWindows()

