import cv2
import face_recognition
import numpy as np 

# โหลดและเข้ารหัสใบหน้าที่รู้จัก
known_face_encodings = []
known_face_names = []

data = {
    "Kan": r"D:\workinghard\Python\data\Kan\kan1.png",
    "Mark": r"D:\workinghard\Python\data\mark\mark1.jpg",
    "Toon": r"D:\workinghard\Python\data\toon\toon1.jpg",
    "Wuna": r"D:\workinghard\Python\data\wuna\wuna.jpg",
    "Pound": r"D:\workinghard\Python\data\pon\pon1.jpg",
    "Ball": r"D:\workinghard\Python\data\ball\ball1.jpg",
}

for name, path in data.items():
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:  # ตรวจสอบว่าพบใบหน้าในภาพไหม
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

# เปิดกล้อง
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    # **ลดขนาดภาพให้เล็กลงเพื่อประมวลผลเร็วขึ้น**
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # ย่อ 50%
    
    # หาตำแหน่งใบหน้า
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)  # หาค่าที่ใกล้เคียงที่สุด

        if face_distances[best_match_index] < 0.5:  # ค่าต่ำกว่า 0.5 คือ match
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        
        # ขยายค่าตำแหน่งกลับมาเป็นขนาดปกติ
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()