import os
import numpy as np
import torch
import librosa
import sounddevice as sd
from models.voice_model import VoiceEmbeddingModel


class VoiceEngine:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "voice_model.pth"
        )

        self.model = VoiceEmbeddingModel()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.sample_rate = 16000
        self.duration = 3
        self.n_mfcc = 40

    def record_audio(self):
        print("Recording... speak now")
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        print("Recording done!")
        return audio.flatten()

    def extract_features(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        mfcc = librosa.util.normalize(mfcc)
        return mfcc

    def extract_embedding(self):
        audio = self.record_audio()

        if audio is None or len(audio) == 0:
            print("Audio recording failed.")
            return None

        rms = np.sqrt(np.mean(audio ** 2))
        print(f"Audio RMS: {rms:.6f}")

        if rms < 0.01:
            print("No real speech detected.")
            return None

        mfcc = self.extract_features(audio)

        mfcc_tensor = torch.FloatTensor(mfcc)
        mfcc_tensor = mfcc_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(mfcc_tensor)

        return embedding

    def compare(self, embedding, voice_matrix, user_ids):
        import faiss

        dimension = 128
        index = faiss.IndexFlatIP(dimension)

        voice_matrix = np.asarray(voice_matrix, dtype=np.float32)
        if voice_matrix.ndim == 1:
            voice_matrix = voice_matrix.reshape(1, -1)

        faiss.normalize_L2(voice_matrix)
        index.add(voice_matrix)

        query = embedding.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query)

        distances, indices = index.search(query, k=1)

        confidence = float(distances[0][0])
        match_idx = int(indices[0][0])
        match_id = user_ids[match_idx]

        return match_id, confidence