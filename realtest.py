import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
face = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(frame,
                                  scaleFactor=1.1,
                                  minNeighbors=5,
                                  minSize=(5, 5))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 显示结果帧e
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()
