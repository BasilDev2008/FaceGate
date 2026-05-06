import os
import cv2
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from models.face_model import FaceEmbeddingModel
from models.voice_model import VoiceEmbeddingModel


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# This is the SAME folder register.py writes photos into:
FACE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "faces")
VOICE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "voices")

# This is the SAME folder face_engine.py loads the model from:
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_model.pth")
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_model.pth")


# =========================
# FACE DATASET
# =========================

class FaceDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Face data folder not found: {data_folder}")

        people = [
            person for person in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, person))
        ]

        people.sort()

        if len(people) < 2:
            raise ValueError(
                "Need at least 2 people folders inside data/faces for triplet training."
            )

        self.label_map = {person: idx for idx, person in enumerate(people)}

        for person in people:
            person_folder = os.path.join(data_folder, person)

            image_count = 0

            for image_file in os.listdir(person_folder):
                full_path = os.path.join(person_folder, image_file)

                if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.data.append((full_path, self.label_map[person]))
                    image_count += 1

            print(f"{person}: {image_count} images")

        if len(self.data) == 0:
            raise ValueError("No face images found.")

        print("\nFace dataset loaded.")
        print("Face data folder:", data_folder)
        print("People:", self.label_map)
        print("Total face images:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# =========================
# VOICE DATASET
# =========================

class VoiceDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        self.sample_rate = 16000
        self.n_mfcc = 40

        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Voice data folder not found: {data_folder}")

        people = [
            person for person in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, person))
        ]

        people.sort()

        if len(people) < 2:
            raise ValueError("Need at least 2 people folders for voice triplet training.")

        self.label_map = {person: idx for idx, person in enumerate(people)}

        for person in people:
            person_folder = os.path.join(data_folder, person)

            for recording_file in os.listdir(person_folder):
                full_path = os.path.join(person_folder, recording_file)

                if recording_file.lower().endswith(".npy"):
                    self.data.append((full_path, self.label_map[person]))

        if len(self.data) == 0:
            raise ValueError("No voice recordings found.")

        print("Voice dataset loaded.")
        print("People:", self.label_map)
        print("Total voice samples:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recording_path, label = self.data[idx]

        audio = np.load(recording_path)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )

        mfcc = librosa.util.normalize(mfcc)
        mfcc = torch.FloatTensor(mfcc)

        return mfcc, torch.tensor(label, dtype=torch.long)


# =========================
# TRIPLET LOSS
# =========================

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)

        loss = torch.clamp(
            positive_distance - negative_distance + self.margin,
            min=0.0
        )

        return loss.mean()


# =========================
# TRIPLET CREATION
# =========================

def get_triplets(embeddings, labels):
    """
    Creates anchor-positive-negative triplets inside the batch.

    Do NOT detach embeddings.
    Detaching kills learning.
    """

    triplets = []

    labels_np = labels.detach().cpu().numpy()

    for idx in range(len(embeddings)):
        anchor_label = labels_np[idx]

        positive_indices = np.where(labels_np == anchor_label)[0]
        positive_indices = positive_indices[positive_indices != idx]

        negative_indices = np.where(labels_np != anchor_label)[0]

        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue

        positive_idx = np.random.choice(positive_indices)
        negative_idx = np.random.choice(negative_indices)

        anchor = embeddings[idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]

        triplets.append((anchor, positive, negative))

    return triplets


# =========================
# TRAINING FUNCTION
# =========================

def train_model(model, dataset, epochs=50, batch_size=32, learning_rate=0.001, model_name="model"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = TripletLoss(margin=0.5)

    model.train()

    print(f"\nTraining {model_name}...")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        batches_used = 0
        triplets_used = 0

        for batch_data, batch_labels in dataloader:
            embeddings = model(batch_data)

            triplets = get_triplets(embeddings, batch_labels)

            if len(triplets) == 0:
                continue

            anchors = torch.stack([t[0] for t in triplets])
            positives = torch.stack([t[1] for t in triplets])
            negatives = torch.stack([t[2] for t in triplets])

            loss = criterion(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches_used += 1
            triplets_used += len(triplets)

        avg_loss = total_loss / max(batches_used, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"Batches used: {batches_used} | "
            f"Triplets used: {triplets_used}"
        )

        if triplets_used == 0:
            print("WARNING: No triplets were created this epoch.")
            print("You need multiple images per person and multiple people in each batch.")

    return model


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("FaceGate Training")
    print("-----------------")
    print("Face data path:", FACE_DATA_DIR)
    print("Voice data path:", VOICE_DATA_DIR)
    print("Model save folder:", MODEL_DIR)

    # -------- FACE MODEL --------

    print("\nLoading face dataset...")
    face_dataset = FaceDataset(FACE_DATA_DIR)

    face_model = FaceEmbeddingModel()

    face_model = train_model(
        model=face_model,
        dataset=face_dataset,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        model_name="face model"
    )

    torch.save(face_model.state_dict(), FACE_MODEL_PATH)

    print("\nFace model saved to:", FACE_MODEL_PATH)

    # -------- VOICE MODEL --------
    # Voice is optional for now.

    try:
        print("\nLoading voice dataset...")
        voice_dataset = VoiceDataset(VOICE_DATA_DIR)

        voice_model = VoiceEmbeddingModel()

        voice_model = train_model(
            model=voice_model,
            dataset=voice_dataset,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            model_name="voice model"
        )

        torch.save(voice_model.state_dict(), VOICE_MODEL_PATH)

        print("Voice model saved to:", VOICE_MODEL_PATH)

    except Exception as error:
        print("\nVoice training skipped.")
        print("Reason:", error)