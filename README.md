# Face Recognition Attendance System

A real-time face recognition attendance system that detects faces using a webcam and logs the recognized users into a Google Sheet automatically.

## Features
- **Real-time Face Detection**: Uses OpenCV and face_recognition library to identify faces in a video stream.
- **Google Sheets Integration**: Automatically records attendance with timestamps.
- **Optimized Performance**: Downscales frames to improve processing speed.
- **Duplicate Prevention**: Avoids logging the same person multiple times within a short time frame.

## Prerequisites
Ensure you have the following installed before running the project:

- Python 3.x
- OpenCV (`cv2`)
- `face_recognition` library
- `numpy`
- `gspread` (Google Sheets API)
- Google Service Account Credentials (JSON file)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/face-recognition-attendance.git
   cd face-recognition-attendance
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Setup Google Sheets API:
   - Create a service account in Google Cloud.
   - Download the JSON credentials file and place it in the project directory.
   - Share the Google Sheet with the service account email.
4. Modify `credentials.json` and `sheet_id` in the script accordingly.
5. Add known faces:
   - Store images in an appropriate folder and update the `data` dictionary in the script with the correct paths.


## Customization
- Modify the `data` dictionary to include more people.
- Adjust the `face_recognition.face_distance` threshold to fine-tune recognition accuracy.
- Switch from `hog` to `cnn` model for better performance (requires GPU).



