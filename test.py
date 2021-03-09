import cv2

## 加载分类器 cv2.CascadeClassifier
'''
CascadeClassifier是Opencv中做人脸检测时候的一个级联分类器，
该类中封装的是目标检测机制即滑动窗口机制+级联分类器的方式。
数据结构包括Data和FeatureEvaluator两个主要部分。
Data中存储的是从训练获得的xml文件中载入的分类器数据；
而FeatureEvaluator中是关于特征的载入、存储和计算。
这里采用的训练文件是OpenCV中默认提供的haarcascade_frontalface_default.xml。
至于Haar，LBP的具体原理，可以参考opencv的相关文档，简单地，可以理解为人脸的特征数据。
'''
face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

img = cv2.imread('1-1.jpg')

## 多尺度检测 detectMultiScale
'''
调用 CascadeClassifier 中的调detectMultiScale函数进行多尺度检测，多尺度检测中会调用单尺度的方法detectSingleScale。

参数说明：
scaleFactor 是 图像的缩放因子
minNeighbors 为每一个级联矩形应该保留的邻近个数，可以理解为一个人周边有几个人脸
minSize 是检测窗口的大小

这些参数都是可以针对图片进行调整的，处理结果返回一个人脸的矩形对象列表。
'''
faces = face.detectMultiScale(img,
                              scaleFactor=1.1,
                              minNeighbors=5,
                              minSize=(5, 5))

## 为每个人脸画一个框
'''
循环读取人脸的矩形对象列表，获得人脸矩形的坐标和宽高，
 然后在原图片中画出该矩形框，调用的是OpenCV的rectangle 方法，其中矩形框的颜色等是可调整的。
'''
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# cv2.imwrite('result.png', img)
cv2.imshow('pic', img)
cv2.waitKey()