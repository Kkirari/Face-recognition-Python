import cv2
import face_recognition
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
client = gspread.authorize(creds)

sheet_id = "1zocAFu-uHxQj0qThRIatVUtFdqKLgo4oXJ0E8xfDhds"
workbook = client.open_by_key(sheet_id)


new_worksheet_name = "Attendance"
worksheet_list = [ws.title for ws in workbook.worksheets()]

if new_worksheet_name in worksheet_list:
    sheet = workbook.worksheet(new_worksheet_name)
else:
    sheet = workbook.add_worksheet(new_worksheet_name, rows=100, cols=10)
    sheet.append_row(["Timestamp", "Name"]) 


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


video_capture = cv2.VideoCapture(0)

recently_logged = {}  

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue


    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


    face_locations = face_recognition.face_locations(small_frame, model="hog")  # เปลี่ยนเป็น "cnn" หากใช้ GPU
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    detected_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.5: 
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        
        detected_names.append(name)

        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for name in detected_names:
        if name != "Unknown" and (name not in recently_logged or (datetime.now() - recently_logged[name]).seconds > 10):
            sheet.append_row([now, name])
            recently_logged[name] = datetime.now()  

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
