from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os
import sys
import base64
import cv2
import numpy as np


# Allow imports from backend folder
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from face_engine import FaceEngine
from database import Database


router = APIRouter()

face_engine = FaceEngine()
db = Database()


class FrameRequest(BaseModel):
    image: str


def normalize_embedding(embedding):
    """
    Converts embedding to normalized NumPy array shape (1, 128).
    Handles both PyTorch tensors and NumPy arrays.
    """

    if embedding is None:
        return None

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()

    embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norm, 1e-12, None)

    return embedding.astype(np.float32)


def cosine_similarity(a, b):
    a = normalize_embedding(a)
    b = normalize_embedding(b)

    if a is None or b is None:
        return -1.0

    return float(np.dot(a, b.T)[0][0])


def fix_box(box):
    """
    Converts box into x, y, w, h.
    Expected from face_engine: (x, y, w, h)
    """

    if box is None:
        return None

    box = np.array(box).flatten()

    if len(box) < 4:
        return None

    x, y, w, h = box[:4]

    return int(x), int(y), int(w), int(h)


def recognize_single_face(live_embedding, users):
    """
    Compares one live face embedding against all registered users.
    Returns name, confidence, recognized.
    """

    if not users:
        return "Unknown", 0.0, False

    live_embedding = normalize_embedding(live_embedding)

    if live_embedding is None:
        return "Unknown", 0.0, False

    scores = []

    for user in users:
        try:
            stored_face_embedding, _ = db.get_embedding(user)
            stored_face_embedding = normalize_embedding(stored_face_embedding)

            score = cosine_similarity(live_embedding, stored_face_embedding)
            scores.append((user, score))

            print(f"Compared with {user.name}: {score:.4f}")

        except Exception as error:
            print(f"Could not compare with user {user.name}: {error}")

    if not scores:
        return "Unknown", 0.0, False

    scores.sort(key=lambda item: item[1], reverse=True)

    best_user, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    print("BEST MATCH:", best_user.name)
    print("BEST SCORE:", best_score)
    print("SECOND SCORE:", second_score)
    print("MARGIN:", margin)

    # Normal demo-safe values.
    # If correct user shows Unknown, lower FACE_THRESHOLD slightly.
    # If wrong names appear, raise MARGIN_THRESHOLD slightly.
    FACE_THRESHOLD = 0.15
    MARGIN_THRESHOLD = 0.05

    if best_score < FACE_THRESHOLD:
        return "Unknown", best_score, False

    if len(scores) > 1 and margin < MARGIN_THRESHOLD:
        return "Unknown", best_score, False

    return best_user.name, best_score, True


@router.post("/recognize-frame")
async def recognize_frame(request: FrameRequest):
    try:
        image_data = request.image.split(",")[-1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        frame_height, frame_width = frame.shape[:2]

        users = db.get_all_users()

        detected_faces = face_engine.extract_all_embeddings(frame)

        results = []

        for live_embedding, box in detected_faces:
            fixed_box = fix_box(box)

            if fixed_box is None:
                continue

            name, confidence, recognized = recognize_single_face(
                live_embedding=live_embedding,
                users=users
            )

            x, y, w, h = fixed_box

            results.append({
                "name": name,
                "confidence": float(confidence),
                "recognized": bool(recognized),
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }
            })

        return {
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "faces": results
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))