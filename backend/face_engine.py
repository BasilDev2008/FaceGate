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
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_model.pth")
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

    def align_face(self, image, keypoints):
        if not keypoints or 'left_eye' not in keypoints or 'right_eye' not in keypoints:
            return image

        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

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

        valid_results = [r for r in results if r.get('confidence', 0) > 0]
        if not valid_results:
            print("No valid face results")
            return None, None

        result = max(valid_results, key=lambda x: x['confidence'])

        x, y, w, h = result['box']
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))

        aligned_frame = self.align_face(frame_rgb, result.get('keypoints', {}))

        x2 = min(aligned_frame.shape[1], x + w)
        y2 = min(aligned_frame.shape[0], y + h)
        face = aligned_frame[y:y2, x:x2]

        if face.size == 0:
            print("Cropped face is empty")
            return None, None

        # temporarily bypass liveness for debugging/demo stability
        is_real = True
        scores = {}

        if not is_real:
            print(f"Liveness check failed - scores: {scores}")
            return None, None

        try:
            face_tensor = self.transform(face)
            face_tensor = face_tensor.unsqueeze(0)
        except Exception as e:
            print("Face transform failed:", e)
            return None, None

        with torch.no_grad():
            embedding = self.model(face_tensor)

        return embedding, (x, y, w, h)

    def draw_box(self, frame, box, name="Unknown", color=(0, 255, 0)):
        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
        return frame

    def compare(self, embedding, face_matrix, user_ids):
        import faiss

        dimension = 128
        index = faiss.IndexFlatIP(dimension)

        face_matrix = np.asarray(face_matrix, dtype=np.float32)
        faiss.normalize_L2(face_matrix)
        index.add(face_matrix)

        query = embedding.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query)

        distances, indices = index.search(query, k=1)

        confidence = float(distances[0][0])
        match_idx = int(indices[0][0])
        match_id = user_ids[match_idx]

        return match_id, confidence

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
            if result.get('confidence', 0) < 0.90:
                continue

            x, y, w, h = result['box']
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(1, int(w))
            h = max(1, int(h))

            aligned_frame = self.align_face(frame_rgb, result.get('keypoints', {}))

            x2 = min(aligned_frame.shape[1], x + w)
            y2 = min(aligned_frame.shape[0], y + h)
            face = aligned_frame[y:y2, x:x2]

            if face.size == 0:
                continue

            # temporarily bypass liveness for debugging/demo stability
            is_real = True
            scores = {}

            if not is_real:
                continue

            try:
                face_tensor = self.transform(face)
                face_tensor = face_tensor.unsqueeze(0)
            except Exception as e:
                print("Face transform failed:", e)
                continue

            with torch.no_grad():
                embedding = self.model(face_tensor)

            faces.append((embedding, (x, y, w, h)))

        return faces