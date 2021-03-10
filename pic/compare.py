import face_recognition
from face_recognition.api import face_encodings

n = str(3)
f1 = face_recognition.load_image_file(n + '-1.jpg')
unknown = face_recognition.load_image_file(n + '-2.jpg')
f2 = face_recognition.load_image_file(n + '-3.jpg')

f1_encoding = face_recognition.face_encodings(f1)[0]
unknown_encoding = face_recognition.face_encodings(unknown)[0]
f2_encoding = face_recognition.face_encodings(f2)[0]

print(f1_encoding)
print(unknown_encoding)
print(f2_encoding)
print(
    face_recognition.compare_faces([f1_encoding, f2_encoding],
                                   unknown_encoding))
