import cv2  # for webcam
import sounddevice as sd  # for microphone
import numpy as np  # for math
import os  # for creating folders
import time  # for delays
from mtcnn import MTCNN  # for face detection
def collect_face_data(person_name):
    folder = f"../data/faces/{person_name}" # path to save faces
    os.makedirs(folder, exist_ok=True) #creates a folder
    detector = MTCNN() # loading the face detector
    cap = cv2.VideoCapture(0) # open webcam
    count = 0 # capturing photos for training
    total = 100 # 100 total photos for training data
    print(f"Collecting face data for {person_name}")
    print("Look at the camera. Move your head slightly between captures.")
    while count < total:
        ret, frame = cap.read() # reading the frame from webcam
        if not ret:
            print("Could not read frame")
            break
        results = detector.detect_faces(frame)
        if results:
            result = max(results, key = lambda x: x['confidence']) # most confidence of facial recognitionn
            if result['confidence'] > 0.95:
                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y) #ensures non negative coordinates
                face = frame[y:y+h, x:x+w] # takes only the face from the frame
                if face.size > 0:
                    filename = f"{folder}/{person_name}_{count}.jpg"
                    cv2.imwrite(filename, face) #write face to folder
                    count+=1
                    print(f"Captured {count} / {total}")
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{count}/{total}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # above lines place a rectangular box around the face so user is able to see what the camera is capturing
                    time.sleep(0.5)  # half a second between captures
                cv2.imshow("Collecteing Face Data - Press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cap.release() # closes webcam
                cv2.destroyAllWindows()
                print(f"Face data collection complete. Saved {count} photos to {folder}")