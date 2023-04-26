import os

import cv2
import numpy as np

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Get all sub directories in the "train" directory
train_dir = "train"
labels_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# Create a dictionary to map labels to integers
label_map = {label: index for index, label in enumerate(labels_names)}

# Collect all the images in the "train" directory and their labels
images = []
labels = []
for label in label_map.keys():
    for image_path in os.listdir(os.path.join(train_dir, label)):
        image = cv2.imread(os.path.join(train_dir, label, image_path), cv2.IMREAD_GRAYSCALE)
        images.append(image)
        labels.append(label_map[label])

# Train the recognizer
recognizer.train(images, np.array(labels))

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the face ROI
        roi = gray[y:y + h, x:x + w]

        # Predict the label of the face
        label, confidence = recognizer.predict(roi)

        # Draw the label on the frame
        cv2.putText(frame, str(labels_names[label]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
