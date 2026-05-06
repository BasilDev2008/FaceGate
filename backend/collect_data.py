import cv2
import os
import time
from mtcnn import MTCNN


def collect_face_data(person_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "..", "data", "faces", person_name)
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    print("Loading MTCNN...")
    detector = MTCNN()
    print("MTCNN loaded!")

    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("Camera opened!")
    print(f"Collecting face data for {person_name}")
    print("Look at the camera and move your head slightly between captures.")
    print(f"Saving images to: {folder}")

    count = 0
    total = 100
    last_capture_time = 0

    while count < total:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb)

        if results:
            result = max(results, key=lambda x: x["confidence"])

            if result["confidence"] > 0.95:
                x, y, w, h = result["box"]
                x = max(0, int(x))
                y = max(0, int(y))
                w = max(1, int(w))
                h = max(1, int(h))

                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                face = frame[y:y2, x:x2]

                if face.size > 0:
                    current_time = time.time()

                    if current_time - last_capture_time >= 0.5:
                        filename = os.path.join(folder, f"{person_name}_{count}.jpg")
                        cv2.imwrite(filename, face)
                        count += 1
                        last_capture_time = current_time
                        print(f"Captured {count}/{total}")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{count}/{total}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Collecting Face Data - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Face data collection complete. Saved {count} photos to {folder}")


if __name__ == "__main__":
    name = input("Enter person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
    else:
        collect_face_data(name)