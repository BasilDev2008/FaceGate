import cv2
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from mtcnn import MTCNN
from models.face_model import FaceEmbeddingModel
from liveness import LivenessDetector


class FaceEngine:
    def __init__(self):
        self.liveness = LivenessDetector()
        self.detector = MTCNN()

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # IMPORTANT:
        # Since face_engine.py is in backend/
        # and face_model.py is imported from backend/models/,
        # the weights should also be in backend/models/.
        model_path = os.path.join(base_dir, "models", "face_model.pth")

        print("FACE MODEL PATH:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model not found at: {model_path}")

        self.model = FaceEmbeddingModel()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def normalize_embedding(self, embedding):
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.clip(norm, 1e-12, None)

        return embedding.astype(np.float32)

    def align_face(self, image, keypoints):
        if not keypoints or "left_eye" not in keypoints or "right_eye" not in keypoints:
            return image

        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]

        left_x = float(left_eye[0])
        left_y = float(left_eye[1])
        right_x = float(right_eye[0])
        right_y = float(right_eye[1])

        dx = right_x - left_x
        dy = right_y - left_y
        angle = np.degrees(np.arctan2(dy, dx))

        eye_center = (
            float((left_x + right_x) / 2.0),
            float((left_y + right_y) / 2.0)
        )

        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        return aligned

    def crop_face(self, frame_rgb, result):
        x, y, w, h = result["box"]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        img_h, img_w = frame_rgb.shape[:2]

        # Add padding so the crop is not too tight
        pad_x = int(w * 0.20)
        pad_y = int(h * 0.25)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None, None

        face = frame_rgb[y1:y2, x1:x2]

        return face, (x, y, w, h)

    def extract_embedding(self, frame):
        if frame is None:
            print("Frame is None")
            return None, None

        if not isinstance(frame, np.ndarray):
            print(f"Invalid frame type: {type(frame)}")
            return None, None

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Invalid frame shape: {frame.shape}")
            return None, None

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)
        except Exception as e:
            print("MTCNN detect_faces crashed:", e)
            return None, None

        if not results:
            return None, None

        valid_results = [
            r for r in results
            if r.get("confidence", 0) >= 0.90
        ]

        if not valid_results:
            return None, None

        result = max(valid_results, key=lambda x: x["confidence"])

        # Align full frame first
        aligned_frame = self.align_face(frame_rgb, result.get("keypoints", {}))

        face, box = self.crop_face(aligned_frame, result)

        if face is None or face.size == 0:
            print("Cropped face is empty")
            return None, None

        try:
            face_tensor = self.transform(face)
            face_tensor = face_tensor.unsqueeze(0)
        except Exception as e:
            print("Face transform failed:", e)
            return None, None

        with torch.no_grad():
            embedding = self.model(face_tensor)

        embedding_np = self.normalize_embedding(embedding)

        print("Embedding shape:", embedding_np.shape)
        print("Embedding norm:", np.linalg.norm(embedding_np))
        print("Embedding first 5:", embedding_np[0][:5])

        return embedding_np, box

    def extract_all_embeddings(self, frame):
        if frame is None:
            print("Frame is None")
            return []

        if not isinstance(frame, np.ndarray):
            print(f"Invalid frame type: {type(frame)}")
            return []

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Invalid frame shape: {frame.shape}")
            return []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(frame_rgb)
        except Exception as e:
            print("MTCNN detect_faces crashed:", e)
            return []

        if not results:
            return []

        faces = []

        for result in results:
            if result.get("confidence", 0) < 0.90:
                continue

            aligned_frame = self.align_face(frame_rgb, result.get("keypoints", {}))

            face, box = self.crop_face(aligned_frame, result)

            if face is None or face.size == 0:
                continue

            try:
                face_tensor = self.transform(face)
                face_tensor = face_tensor.unsqueeze(0)
            except Exception as e:
                print("Face transform failed:", e)
                continue

            with torch.no_grad():
                embedding = self.model(face_tensor)

            embedding_np = self.normalize_embedding(embedding)

            faces.append((embedding_np, box))

        return faces

    def compare(self, embedding, face_matrix, user_ids):
        embedding = self.normalize_embedding(embedding)
        face_matrix = np.asarray(face_matrix, dtype=np.float32)

        norms = np.linalg.norm(face_matrix, axis=1, keepdims=True)
        face_matrix = face_matrix / np.clip(norms, 1e-12, None)

        similarities = np.dot(face_matrix, embedding.T).flatten()

        best_index = int(np.argmax(similarities))
        confidence = float(similarities[best_index])
        match_id = user_ids[best_index]

        return match_id, confidence

    def draw_box(self, frame, box, name="Unknown", color=(0, 255, 0)):
        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        return frame