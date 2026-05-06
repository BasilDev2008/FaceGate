from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os
import sys
import cv2
import numpy as np
import uuid
import time
import re

# Let this file import backend-level files even though it is inside routes/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from face_engine import FaceEngine
from database import Database


router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    role: str


def safe_folder_name(name):
    """
    Converts 'Basil Loubani' into 'Basil_Loubani'
    and removes unsafe filename characters.
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name


def fix_box(box, frame_width, frame_height):
    """
    Converts detected_box into x, y, w, h format.

    Supports:
    1. [x, y, w, h]
    2. [x1, y1, x2, y2]
    """

    if box is None:
        return None

    box = np.array(box).flatten()

    if len(box) < 4:
        return None

    a, b, c, d = box[:4]
    a, b, c, d = int(a), int(b), int(c), int(d)

    candidates = []

    # Candidate 1: x, y, w, h
    x, y, w, h = a, b, c, d
    if w > 0 and h > 0:
        candidates.append((x, y, w, h))

    # Candidate 2: x1, y1, x2, y2
    x1, y1, x2, y2 = a, b, c, d
    if x2 > x1 and y2 > y1:
        candidates.append((x1, y1, x2 - x1, y2 - y1))

    if not candidates:
        return None

    best_box = None
    best_score = -1

    for x, y, w, h in candidates:
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))

        area_ratio = (w * h) / (frame_width * frame_height)

        if area_ratio < 0.01 or area_ratio > 0.80:
            continue

        score = 1 - abs(area_ratio - 0.15)

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box


def open_camera():
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"Camera opened with index {camera_index}.")
            return cap

        cap.release()

    return None


def to_numpy_embedding(embedding):
    """
    Handles both possible outputs:
    - PyTorch tensor
    - NumPy array

    Returns shape: (1, 128)
    """

    if embedding is None:
        return None

    if hasattr(embedding, "detach"):
        embedding_np = embedding.detach().cpu().numpy().astype(np.float32)
    else:
        embedding_np = np.asarray(embedding, dtype=np.float32)

    embedding_np = embedding_np.reshape(1, -1).astype(np.float32)

    norm = np.linalg.norm(embedding_np, axis=1, keepdims=True)
    embedding_np = embedding_np / np.clip(norm, 1e-12, None)

    return embedding_np.astype(np.float32)


def save_training_image(frame, fixed_box, save_dir, safe_name, sample_number):
    """
    Saves the detected face crop into data/faces/Person_Name/.
    This is what train.py will later learn from.
    """

    x, y, w, h = fixed_box

    frame_height, frame_width = frame.shape[:2]

    # Add padding around face so training images are not too tight.
    pad_x = int(w * 0.20)
    pad_y = int(h * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        print("Skipped saving empty face crop.")
        return None

    timestamp = int(time.time() * 1000)
    image_filename = f"{safe_name}_{timestamp}_{sample_number}.jpg"
    image_path = os.path.join(save_dir, image_filename)

    cv2.imwrite(image_path, face_crop)

    return image_path


def collect_face_embedding(name):
    face_engine = FaceEngine()

    cap = open_camera()

    if cap is None:
        raise RuntimeError("Could not open camera")

    safe_name = safe_folder_name(name)

    # This is the exact folder train.py will read from.
    face_data_dir = os.path.join(PROJECT_ROOT, "data", "faces", safe_name)
    os.makedirs(face_data_dir, exist_ok=True)

    print("Saving training face images to:", face_data_dir)

    collected_embeddings = []
    target_samples = 30

    capture_started = False
    last_capture_time = 0
    capture_delay = 0.5

    window_name = "FaceGate Register"

    button_state = {
        "start_clicked": False,
        "quit_clicked": False
    }

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # START button
            if 20 <= x <= 180 and 140 <= y <= 190:
                button_state["start_clicked"] = True

            # QUIT button
            elif 200 <= x <= 360 and 140 <= y <= 190:
                button_state["quit_clicked"] = True

            # Click anywhere else also starts
            else:
                button_state["start_clicked"] = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Camera window opened.")
    print("Wait for the green face box.")
    print("Click START to capture samples.")
    print("Click QUIT to cancel.")

    try:
        while len(collected_embeddings) < target_samples:
            ret, frame = cap.read()

            if not ret or frame is None:
                raise RuntimeError("Could not read camera frame")

            frame_height, frame_width = frame.shape[:2]

            embedding, detected_box = face_engine.extract_embedding(frame)

            fixed_box = fix_box(detected_box, frame_width, frame_height)
            embedding_np = to_numpy_embedding(embedding)

            face_detected = fixed_box is not None and embedding_np is not None

            if fixed_box is not None:
                x, y, w, h = fixed_box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

            status_text = "FACE DETECTED" if face_detected else "NO FACE"
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                2
            )

            cv2.putText(
                frame,
                f"Samples: {len(collected_embeddings)}/{target_samples}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            if capture_started:
                instruction_text = "Capturing... slowly move head"
                instruction_color = (0, 255, 0)
            else:
                instruction_text = "Click START when face is detected"
                instruction_color = (255, 255, 255)

            cv2.putText(
                frame,
                instruction_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                instruction_color,
                2
            )

            # START button
            cv2.rectangle(frame, (20, 140), (180, 190), (0, 180, 0), -1)
            cv2.putText(
                frame,
                "START",
                (50, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            # QUIT button
            cv2.rectangle(frame, (200, 140), (360, 190), (0, 0, 180), -1)
            cv2.putText(
                frame,
                "QUIT",
                (240, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

            if button_state["quit_clicked"]:
                raise RuntimeError("Registration cancelled")

            if button_state["start_clicked"]:
                button_state["start_clicked"] = False

                if not face_detected:
                    print("No valid face detected. Wait for the green box.")
                else:
                    capture_started = True
                    print("Capture started.")

            if capture_started:
                if face_detected:
                    current_time = time.time()

                    if current_time - last_capture_time >= capture_delay:
                        collected_embeddings.append(embedding_np)
                        last_capture_time = current_time

                        sample_number = len(collected_embeddings)

                        saved_path = save_training_image(
                            frame=frame,
                            fixed_box=fixed_box,
                            save_dir=face_data_dir,
                            safe_name=safe_name,
                            sample_number=sample_number
                        )

                        if saved_path:
                            print("Saved training image:", saved_path)

                        print(f"Captured sample {sample_number}/{target_samples}")
                else:
                    print("Face lost. Waiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(collected_embeddings) < target_samples:
        raise RuntimeError("Registration failed. Not enough samples")

    face_stack = np.vstack(collected_embeddings).astype(np.float32)

    mean_face_embedding = np.mean(face_stack, axis=0, keepdims=True)

    norm = np.linalg.norm(mean_face_embedding, axis=1, keepdims=True)
    mean_face_embedding = mean_face_embedding / np.clip(norm, 1e-12, None)

    mean_face_embedding = mean_face_embedding.astype(np.float32)

    print("Final face embedding shape:", mean_face_embedding.shape)
    print("Final face embedding norm:", np.linalg.norm(mean_face_embedding))
    print("Final face embedding first 5:", mean_face_embedding[0][:5])

    return mean_face_embedding


def save_user(name, role):
    db = Database()

    face_embedding = collect_face_embedding(name)

    # Your database.py has voice_embedding nullable=False.
    # So we give it a dummy 128-dim vector.
    dummy_voice_embedding = np.zeros((1, 128), dtype=np.float32)

    user_id = str(uuid.uuid4())

    db.add_user(
        id=user_id,
        name=name,
        role=role,
        face_embedding=face_embedding,
        voice_embedding=dummy_voice_embedding
    )

    print("SUCCESS: Face registered and saved.")
    print("User ID:", user_id)
    print("Name:", name)
    print("Role:", role)
    print("Face embedding shape sent to DB:", face_embedding.shape)
    print("Dummy voice embedding shape sent to DB:", dummy_voice_embedding.shape)

    return {
        "message": "Face registered successfully",
        "user_id": user_id,
        "name": name,
        "role": role,
        "face_embedding_shape": list(face_embedding.shape),
        "voice_embedding_shape": list(dummy_voice_embedding.shape)
    }


@router.post("/register")
def register(request: RegisterRequest):
    name = request.name.strip()
    role = request.role.strip()

    if name == "":
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    if role == "":
        raise HTTPException(status_code=400, detail="Role cannot be empty")

    try:
        return save_user(name, role)

    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(error)}")


def run_manual_register():
    print("FaceGate Manual Registration")
    print("----------------------------")

    name = input("Enter name: ").strip()
    role = input("Enter role: ").strip()

    if name == "":
        print("Name cannot be empty.")
        return

    if role == "":
        print("Role cannot be empty.")
        return

    try:
        result = save_user(name, role)
        print("Registration result:")
        print(result)

    except Exception as error:
        print("Registration failed:", error)


if __name__ == "__main__":
    run_manual_register()