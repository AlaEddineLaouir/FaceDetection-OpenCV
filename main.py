from __future__ import print_function
import cv2 as cv

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

face_cascade.load(cv.samples.findFile('classifier.xml'))
eyes_cascade.load(cv.samples.findFile('eyeclassifier.xml'))
camInput = cv.VideoCapture(0)

while True:
    booleanCheck, frame = camInput.read()

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

    cv.imshow('Capture - Face detection', frame)
    keyPressed = cv.waitKey(1)
    if keyPressed == ord('s'):
        break



camInput.release()

cv.closeAllWindows()
