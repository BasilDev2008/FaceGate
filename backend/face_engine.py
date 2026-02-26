import cv2 # for webcam and image processing
import numpy as np # for math operations on matricees
import torch # running the model
import math
import torchvision.transforms as transforms # preparing perfect images for the model
from mtcnn import MTCNN # detecting faces and eye coordinates
from models.face_model import FaceEmbeddingModel # model of traning in face_model.py

class FaceEngine:
    def __init__(self):
        self.detector = MTCNN() # loads face detector
        self.model = FaceEmbeddingModel() # loads our face model
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # converts image of pixels RGB values as arrays into a PIL image
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BILINEAR), # resize the PIL image to standard size
            transforms.ToTensor(), # ToTensor helps take PIL image into values used for comparison
            transforms.Normalize( # normalize pixel values
                mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5]
            )
        ])
    def align_face(self, image, keypoints):
        # gets eye coordinations from MTCNN
        left_eye = keypoints['left_eye'] # coordinates x,y of left eye
        right_eye = keypoints['right_eye'] # coordinates of right eye
        dx = right_eye[0] - left_eye[0] # horizontal distance between eyes
        dy = right_eye[1] - left_eye[1] # vertical distance between eyes
        angle = np.degrees(np.arctan2(dy,dx)) # angle in degrees of dy/dx for angle between the two eyes
        eye_center = ((left_eye[0] + right_eye[0])//2,
                      (right_eye[1] + left_eye[1])//2)
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale = 1.0)
        # returns a 2x3 matrix to rotate the image for straightening
        aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        # multiplying each pixel's matrix to the 2x3 matrix to receieve a staright image
        return aligned
    def extract_embedding(self,frame):
        # detecting faces within the frame
        results = self.detector.detect_faces(frame) # list of all detected faces
        if not results: # if not in detected faces
            return None, None
        result = max(results, key = lambda x: x['confident'])
        x, y, w, h = result['box'] # x,y,position,width of face
        x, y = max(0,x), max(0,y) # coordinates non-negative
        aligned_frame = self.align_face(frame, result['keypoints'])
        face = aligned_frame[y:y+h, x:x+w] # crops only face from y point of face to the height of ur face
        if face.size == 0: # cropped face is empty
            return None, None
        # prepare face for the model
        face_tensor = self.transform(face) # applies transformations to the image
        face_tensor = face.unsqueeze(0) 
        with torch.no_grad():
            embedding = self.model(face_tensor) # pass face through recognition model
        return embedding, result['box']
    def draw_box(self, frame, box, name = "Unkown", color = (0,255,0)):
        x, y, w, h = box 
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2) # rectangle around face
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2) # write name above box
        return frame # return frame with box drawn on face
