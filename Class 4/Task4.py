import cv2
import numpy as np
# Load an image and convert it to a NumPy array
image = cv2.imread('smile.jpg') # Replace with your image path
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
# Detect faces in the image
faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1,minNeighbors=5)
# Loop through the detected faces and extract facial features (regions)
for (x, y, w, h) in faces:
    #Slice the image array to extract the face region
    face_region = image[y:y+h, x:x+w]
    # Optional: Display the face region
    cv2.imshow('Face Region', face_region)
    # Extract additional facial features if required (e.g., eyes, nose)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(face_region, scaleFactor=1.1,minNeighbors=5)
for (ex, ey, ew, eh) in eyes:
    eye_region = face_region[ey:ey+eh, ex:ex+ew]
    cv2.imshow('Eye Region', eye_region)
# Show the original image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()