import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    # Neural network component
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        # the above line scans for patterns, edges, shapes, and curves and gives them numbers.
        self.bn = nn.BatchNorm2d(out_channels)
        # scales the numbers found by conv so they don't become too large.
        self.relu = nn.ReLU(inplace = True)
        # if negative, set to 0
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
        # pixel numbers from image go into conv for pattern detection, scaled, negatives removed.
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels), 
            ConvBlock(channels, channels),
            # pattern numbers pass through ConvBlock twice to strengthen patterns.
        )
    def forward(self, x):
        return x + self.block(x)
        # after strengthening and narrowing patterns from ConvBlock, important info might get lost; adding the original value x makes sure nothing gets removed.
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_size = 128):
        # every face representing 128 numbers, a numerical faceprint
        super(FaceEmbeddingModel, self). __init__()
        self.network = nn.Sequential(
            ConvBlock(3, 32, stride = 2), # image will come in RGB, scans all 3 and produces 32 grids, representing different patterns in face, stride being 2 skips every othe pixel
            ConvBlock(32, 64, stride = 2), # takes previous 32 previous grids, and finds 64 more patterns, while using stride to half the grid size
            ResidualBlock(64), # uses previous ResidualBlock to refine the 64 pattern grids, ensuring nothing gets lost
            ConvBlock(64, 128, stride = 2),
            # 128 pattern grids, representing more specific details i.e. eye shapes
            ResidualBlock(128),
            ConvBlock(128, 256, stride = 2),
            ResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1)) # takes all the grids and finds the average of each grid making it simpler
        self.fc = nn.Linear(256, embedding_size)
    def forward(self,x):
        x = self.network(x) # goes through refinement and reassurement through ConvBlock and ResidualBlock
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p = 2, dim = 1)
        # above line normalizes lightening, so face is always assessed equally through different lighting.