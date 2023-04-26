import os

import cv2


# Create a function to detect faces in the video
def detect_faces(video_capt, dir_name):
    # Create a CascadeClassifier object to detect faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i = 0

    while True:
        # Read a frame from the webcam
        ret, frame = video_capt.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.imwrite(f'{dir_name}/face_{i}.jpg', frame[y:y + h, x:x + w])
            i = i + 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the frame on the screen
        cv2.imshow('Webcam', frame)

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capt.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Ask for the person's name
    name = input('Enter the person\'s name: ')

    # Create a directory with the person's name
    os.mkdir(name)

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    # Call the function to detect faces
    detect_faces(video_capture, name)
