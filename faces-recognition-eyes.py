import pickle

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pkl", 'rb') as file:
    og_labels = pickle.load(file)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in faces:
        print(x, y, w, h)

        # Define region of interest
        roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # recognize? deep learned model predict (keras, tensorflow, pytorch, scikit, learn)
        id_, conf = recognizer.predict(roi_gray)
        # if conf>=45 and conf <= 85:
        print(id_)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "N: %s, conf: %d" % (labels[id_].replace("-", " ").title(), conf)
        color = (255, 255, 255) #BGR
        stroke = 2
        cv2.putText(frame, text, (x, y-10), font, 0.5, color, stroke, cv2.LINE_AA)

        # Define image name
        img_item_gray = "my-image-gray.png"
        img_item_color = "my-image-color.png"

        # Write the image to the disk
        cv2.imwrite(img_item_gray, roi_gray)
        cv2.imwrite(img_item_color, roi_color)

        # Set the properties for the rectangle
        rect_color = (255, 0, 0) #BGR 0-255
        rect_stroke = 2

        # Calculate the width for the rectangle
        width = x + w
        height = y + h

        # Create the rectangle
        cv2.rectangle(frame, (x, y), (width, height), rect_color, rect_stroke)

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=8)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
