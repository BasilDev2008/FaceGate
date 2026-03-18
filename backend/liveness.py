import cv2 # image processing
import numpy as np
import torch
import torch.nn as nn
class LivenessModel(nn.Module): # model detecting whether a face is real or fake
    def __init__(self):
        super(LivenessModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # half the image size
            nn.Conv2d(32, 64, kernel_size= 3, padding = 1), # second conv layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # halve image again
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1) ,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # output 2 values - real or fake

        )
    def forward(self, x):
        x = self.network(x) # pass through conv layers
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class LivenessDetector:
    def __init__(self):
        self.model = LivenessModel()
        self.model.eval()
        self.threshold = 0.7
    def check_texture(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = laplacian.var()
        return texture_score
    def check_reflection(self, face):
        # real faces have natural light reflection, photos have flat reflection
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV) # convert to HSV color space
        saturation = hsv[:,:,1] # get saturation channel
        reflection_score = saturation.std() # measure difference in saturation
        return reflection_score
    def check_blur(self, face):
        # printed photos are often slightly blurry around edges
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        # blur_score measures sharpness
        return blur_score # higher score is more real
    def is_live(self, face):
        texture_score = self.check_texture(face)
        reflection_score = self.check_reflection(face)
        blur_score = self.check_blur(face)
        # gets scores from all functions above to make a decision

        texture_pass = texture_score > 100
        reflection_pass = reflection_score > 30
        blur_pass = blur_score > 80
        # texture score greater than 80 passes
        checks_passed = sum([texture_pass, reflection_pass, blur_pass])
        is_real = checks_passed>=2
        return is_real, {
            "texture": texture_score,
            "reflection": reflection_score,
            "blur": blur_score,
            "checks_passed": checks_passed
        }
    