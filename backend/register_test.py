import cv2
import uuid
import numpy as np
from face_engine import FaceEngine
from database import Database


def average_embeddings(embeddings):
    if not embeddings:
        return None

    stacked = np.vstack([emb.detach().cpu().numpy() for emb in embeddings]).astype(np.float32)
    mean_embedding = np.mean(stacked, axis=0, keepdims=True)

    norm = np.linalg.norm(mean_embedding, axis=1, keepdims=True)
    mean_embedding = mean_embedding / np.clip(norm, 1e-12, None)

    return mean_embedding


def collect_face_embeddings(face_engine, target_samples=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    window_name = "FaceGate Face Registration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nFace registration started.")
    print("Click the webcam window first.")
    print(f"Press C {target_samples} times when FACE READY appears.")
    print("Change your angle slightly each time.")
    print("Press Q or ESC to quit.\n")

    collected_embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam.")
            break

        embedding, box = face_engine.extract_embedding(frame)
        display_frame = frame.copy()

        status_text = f"Collected: {len(collected_embeddings)}/{target_samples}"
        status_color = (0, 255, 255)

        if embedding is not None and box is not None:
            display_frame = face_engine.draw_box(display_frame, box, "FACE READY", (0, 255, 0))
            status_text = f"FACE READY - PRESS C ({len(collected_embeddings)}/{target_samples})"
            status_color = (0, 255, 0)

        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.75,
            status_color,
            2
        )

        cv2.putText(
            display_frame,
            "Press C to save sample | Q or ESC to quit",
            (10, 65),
            cv2.FONT_HERSHEY_COMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            print("Face registration cancelled.")
            break

        if key == ord("c"):
            if embedding is None:
                print("Capture blocked: no valid face embedding yet.")
            else:
                collected_embeddings.append(embedding)
                print(f"Face sample {len(collected_embeddings)}/{target_samples} captured.")

                if len(collected_embeddings) >= target_samples:
                    print("Collected all face samples.")
                    break

    cap.release()
    cv2.destroyAllWindows()

    averaged_face = average_embeddings(collected_embeddings)
    return averaged_face


def register_user():
    name = input("Enter name: ").strip()
    role = input("Enter role: ").strip()

    if not name or not role:
        print("Name and role are required.")
        return

    face_engine = FaceEngine()
    db = Database()

    face_embedding = collect_face_embeddings(face_engine, target_samples=5)
    if face_embedding is None:
        print("Face registration failed.")
        return

    user_id = str(uuid.uuid4())

    dummy_voice_embedding = np.zeros((1, 128), dtype=np.float32)

    try:
        db.add_user(
            id=user_id,
            name=name,
            role=role,
            face_embedding=face_embedding,
            voice_embedding=dummy_voice_embedding
        )

        users = db.get_all_users()
        print("USER COUNT AFTER SAVE:", len(users))
        for user in users:
            print(user.id, user.name, user.role)

    except Exception as e:
        print(f"Database save failed: {e}")
        return

    print("\nFace-only registration successful!")
    print(f"Name: {name}")
    print(f"Role: {role}")
    print(f"User ID: {user_id}")
    print(f"Face embedding shape: {face_embedding.shape}")


if __name__ == "__main__":
    register_user()