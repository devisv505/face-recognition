import os

import cv2
import numpy as np

# Create a face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create a face recognizer object
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if the trained model file exists
if os.path.exists("trained_model.yml"):
    # Load the trained model from the file
    face_recognizer.read("trained_model.yml")
    print("Trained model loaded from file.")
else:
    print("Trained model file not found.")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a list to store the labels for the faces
labels = []
# Create a list to store the faces
faces_data = []

while True:
    # Capture the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Append the label and the face to the lists
        labels.append('Denis')
        faces_data.append(gray[y:y + h, x:x + w])

        # Check if the model is computed
        if face_recognizer.getLabels() is not None:
            # Get the label for the face
            label, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])

            # Show the label under the face
            cv2.putText(frame, str(label), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the faces
    cv2.imshow("Webcam", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the window
cap.release()
cv2.destroyAllWindows()

# Train the face recognizer with the captured faces and labels
face_recognizer.train(faces_data, np.array(labels))

# Save the trained model to a file
face_recognizer.save("trained_model.yml")
print("Trained model saved to file.")
