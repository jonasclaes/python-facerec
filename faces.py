import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

    for (x, y, w, h) in faces:
        print("X: %d, Y: %d, W: %d, H: %d." % (x, y, w, h))

        # Define region of interest
        roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

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


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
