import cv2
from face_engine import FaceEngine
from voice_engine import VoiceEngine


def test_face():
    print("Testing face detection...")
    face_engine = FaceEngine()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam.")
            break

        try:
            faces = face_engine.extract_all_embeddings(frame)
        except Exception as e:
            print(f"Face detection crashed: {e}")
            break

        print(f"Faces detected: {len(faces)}", end="\r")

        for embedding, box in faces:
            frame = face_engine.draw_box(frame, box, "Face Detected")

        cv2.imshow("FaceGate Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_voice():
    print("Testing voice detection...")
    voice_engine = VoiceEngine()
    print("Say something...")

    try:
        embedding = voice_engine.extract_embedding()
    except Exception as e:
        print(f"Voice detection crashed: {e}")
        return

    if embedding is not None:
        print(f"Voice detected! Embedding shape: {embedding.shape}")
        print("Voice engine working correctly!")
    else:
        print("No voice detected")


if __name__ == "__main__":
    print("1. Test face detection")
    print("2. Test voice detection")
    choice = input("Choose (1 or 2): ").strip()

    if choice == "1":
        test_face()
    elif choice == "2":
        test_voice()
    else:
        print("Invalid choice. Please enter 1 or 2.")