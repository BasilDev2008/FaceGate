import os
import torch
from models.face_model import FaceEmbeddingModel
from models.voice_model import VoiceEmbeddingModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

face_path = os.path.join(MODELS_DIR, "face_model.pth")
voice_path = os.path.join(MODELS_DIR, "voice_model.pth")

if not os.path.exists(face_path):
    face_model = FaceEmbeddingModel()
    torch.save(face_model.state_dict(), face_path)
    print("Saved face model weights.")
else:
    print("Face model weights already exist.")

if not os.path.exists(voice_path):
    voice_model = VoiceEmbeddingModel()
    torch.save(voice_model.state_dict(), voice_path)
    print("Saved voice model weights.")
else:
    print("Voice model weights already exist.")