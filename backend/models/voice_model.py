import torch 
import torch.nn as nn
import torch.nn.functional as F
class VoiceEmbeddingModel(nn.Module): 
    # converts a voice sample into a unique vector
    def __init__(self, embedding_size = 128):
        super(VoiceEmbeddingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(64), # keeps numbers in healthy range
            nn.ReLU(), # negative numbers are set to 0.
            nn.Conv1d(64, 128, kernel_size = 3, padding = 1), # finding 128 patterns now within voice
            nn.BatchNorm1d(128), # scales numbers again
            nn.ReLU(), # throwing away negatives
            nn.Conv1d(128, 256, kernel_size = 3, padding = 1), # 256 patterns within voices
            nn.BatchNorm1d(256),
            nn.ReLU() 
        )
        self.pool = nn.AdaptiveAvgPool1d(1) # compress everything into a point
        self.fc = nn.Linear(256, embedding_size) # compressing them into 128 numbers
    def forward(self, x):
        x = self.network(x) # voice passes through all layers
        x = self.pool(x) # squash
        x = x.view(x.size(0), -1) # flattens to a 1D grid for voices
        x = self.fc() # voice has a unique imprint
        return F.normalize(x, p = 2, dim = 1) # scales so all voices are within the same range
