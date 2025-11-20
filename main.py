import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
from simple_facerec import SimpleFacerec
from attendance import mark_attendance

# -----------------------------
# Load face encodings
# -----------------------------
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

FRAME_RESIZE = 0.25

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Detect known faces
# -----------------------------
def detect_known_faces(frame_in):
    small_frame = cv2.resize(frame_in, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    known_face_encodings = np.load("ImageEncoding.npy")

    with open("ImageNames.txt", "r") as f:
        known_face_names = [line.strip() for line in f.readlines()]

    detected_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

        detected_names.append(name)

    face_locations = (np.array(face_locations) / FRAME_RESIZE).astype(int)
    return face_locations, detected_names

# -----------------------------
# Main camera function
# -----------------------------
def start_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_names = detect_known_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            face_roi = frame[top:bottom, left:right]

            result = DeepFace.analyze(
                face_roi,
                actions=["emotion"],
                enforce_detection=False
            )
            emotion = result[0]["dominant_emotion"]

            # Draw name
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            # Draw emotion
            cv2.putText(frame, emotion, (left, bottom + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (200, 0, 0), 4)

        cv2.imshow("FaceDetect", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Run the system
# -----------------------------
if _name_ == "_main_":
    start_camera()