import numpy as np
import torch
import librosa
import sounddevice as sd # recording from microphone
from models.voice_model import VoiceEmbeddingModel

class VoiceEngine:
    def __init__(self):
        self.model = VoiceEmbeddingModel() # our voice model
        self.model.eval()
        self.samples_rate = 16000 # 16000 samples per second
        self.duration = 3 # 3 seconds of audio
        self.n_mfcc = 40 # 40 MFCC features of voice from 40,000
        def record_audio(self):
            print("Recording... speak now")
            audio = sd.rec(
                int(self.duration * self.sample_rate), samplerate = self.samplerate,
                channels = 1,
                dtype = 'float32'
            )
            sd.wait()
            print("Recording Done!")
            return audio.flatten() # becomes a 1D array
        def extract_features(self,audio):
            mfcc = librosa.feature.mfcc(
                y = audio,
                sr = self.sample_rate,
                n_mfcc = self.n_mfcc # num of features to extract from voice

            )
            mfcc = librosa.util.normalize(mfcc) # normalize features
            return mfcc
        def extract_embedding(self):
            audio = self.record_audio()
            mfcc = self.extract_features(audio) # extracting 40 features
            mfcc_tensor = torch.FloatTensor(mfcc) # converts numpy array to tensor
            mfcc_tensor = mfcc_tensor.unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(mfcc_tensor) # passes through model
            return embedding # return 128 number voice fingerprint
         
