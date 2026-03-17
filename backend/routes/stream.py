from fastapi import APIRouter # for creating routes
from fastapi.responses import StreamingResponse # for the streaming 
import cv2 # for webcam
import numpy as np
from face_engine import FaceEngine
from voice_engine import VoiceEngine
from database import Database
router = APIRouter()
db = Database()
def get_embeddings():
    users = db.get_all_users() # gets all registered users
    face_embeddings = [] # stores face embeddings
    user_ids = [] # store user ids
    for user in users:
        face_emb, _ = db.get_embedding(user)
        face_embeddings.append(face_emb)
        user_ids.append(user.id)
    if not face_embeddings:
        return None,None
    face_matrix = np.array(face_embeddings, dtype = np.float32) # convert to a matrix
    return face_matrix, user_ids
def generate_frames():
    face_engine = FaceEngine()
    cap = cv2.VideoCapture(0) # opens the webcam
    while True: # infinite
        ret, frame = cap.read() # reads a frame from the webcam
        if not ret:
            break
        face_matrix, user_ids = get_embeddings()
        embedding, box = face_engine.extract_embedding(frame) # detect the face in the frame
        if embedding is not None and face_matrix is not None:
            match_id, confidence = face_engine.compare(
                embedding,
                face_matrix,
                user_ids
            )
            # compare current face with stored embeddings
            if confidence>=0.85:
                user = db.session.query(db.User).filter_by(id = match_id).first() # get user from the database
                name = user.name if user else "Unknown"
                color = (0,255,0)
            else:
                name = "Unknown"
                color = (0,0,255) 
            frame = face_engine.draw_box(frame,box,name,color)
            ret,buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (  # send frame to frontend
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
@router.get("/stream")
async def stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"  # video stream format
    ) 