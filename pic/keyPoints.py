import face_recognition
import cv2

for i in range(1, 9):
    for j in range(1, 4):
        file_name = str(i) + '-' + str(j)
        style = '.jpg'
        img = cv2.imread(file_name + style)
        image = face_recognition.load_image_file(file_name + style)
        face_landmarks_list = face_recognition.face_landmarks(image)
        # print(face_landmarks_list)

        point_size = 4
        point_color = (0, 255, 0)
        for face_landmarks in face_landmarks_list:
            for point in face_landmarks['chin']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['left_eyebrow']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['right_eyebrow']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['nose_bridge']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['nose_tip']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['left_eye']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['right_eye']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['top_lip']:
                d = cv2.circle(img, point, point_size, point_color, -1)
            for point in face_landmarks['bottom_lip']:
                d = cv2.circle(img, point, point_size, point_color, -1)
        cv2.imwrite(file_name + 'kp' + style, img)

# cv2.imshow('pic', img)
# cv2.waitKey()