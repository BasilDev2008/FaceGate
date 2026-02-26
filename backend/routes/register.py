from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from face_engine import FaceEngine
from voice_engine import VoiceEngine
from database import Database
import uuid
import cv2
router = APIRouter()
class RegisterRequest(BaseModel):
    name: str
    role: str
@router.post("/register")
async def register(request: RegisterRequest):
    face_engine = FaceEngine()
    voice_engine = VoiceEngine()
    db = Database()
    cap = cv2.VideoCapture(0)
    face_embedding = None
    box = None
    print("Please look straight at the camera...")
    while True:
        ret, frame = cap.read() # reads a frame from the webcam
        if not ret:
            raise HTTPException(status_code = 500, detail = "Could not access webcam")
        embedding, detected_box = face_engine.extract_embedding(frame)
        if embedding is not None:
            face_embedding = embedding
            box = detected_box
            break
        cap.release()
        if face_embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")
        print("Please speak for the voice recording")
        voice_embedding = voice_engine.extract_embedding()
        if voice_embedding is None:
            raise HTTPException(status_code = 400, detail = "Voice Recording Failed")
        user_id = str(uuid.uuid4())
        db.add_unser(
            id = user_id,
            role = request.role,
            face_embedding=face_embedding,
            voice_embedding = voice_embedding
        )
        return {
            "message": f"{request.name} registered successfully", "user_id": user_id
        }
