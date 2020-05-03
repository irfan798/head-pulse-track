import time

import numpy as np
import cv2
from imutils import face_utils
import dlib


class FacePoints:

    def __init__(self, dedector_type = 'haar'):

        self.dedector_type = dedector_type
        self.init_dedector()

        self.orig_face_rectange = [0,0,0,0]
        self.face_rectange = [0,0,0,0]
        self.eyes_rectangle = [0,0,0,0]

        # parameters of the feature tracking algorithm
        self.feature_params = dict(maxCorners = 500,
                                qualityLevel = 0.01,
                                minDistance = 3,
                                blockSize = 7 )

        

    def init_dedector(self):
        if self.dedector_type == 'haar':
            self.dedector = cv2.CascadeClassifier('./data/haarcascade/haarcascade_frontalface_default.xml')

        elif self.dedector_type == 'dlib':
            # initialize dlib's face detector (HOG-based) and then create
            # the facial landmark predictor
            shape_predict_dat = "./data/shape_predictor_68_face_landmarks.dat"
            self.detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(shape_predict_dat)

        else:
            raise('Not supported dedector')

    def detect_face(self, gray_frame):
        face_rect = None

        if self.dedector_type == 'haar':
            face_rects = self.dedector.detectMultiScale(gray_frame)

        elif self.dedector_type == 'dlib':
            face_rects = self.dedector(gray_frame, 0)
            face_rects = [self.rect_to_bb(face_rect) for face_rect in face_rects ]

        # Get areas of faces
        face_sizes = [h*w for x, y, w, h in face_rects]
        if len(face_rects) > 0:
            # Then get biggest face
            face_idx = np.argmax(face_sizes)
            face_rect = face_rects[face_idx]

        self.orig_face_rectange = face_rect

        return face_rect

    def get_roi_mask(self, gray_frame, face_rectange):
        mask = np.zeros(gray_frame.shape, np.uint8)

        if face_rectange is not None:
            # Get new face rectange
            self.face_rectange = self.resize_face_rectange(*face_rectange)
            # Get eyes
            self.eyes_rectangle = self.remove_eyes_rectangle(*self.face_rectange)

            x,y,w,h = self.face_rectange
            xx,yy,ww,hh = self.eyes_rectangle

            mask[y:y+h,x:x+w] = 255 # Add face
            mask[yy:yy+hh,xx:xx+ww] = 0 # Remove eyes 

        self.mask =  mask
        return mask

    def get_points_pipeline(self, gray_frame):
        face_rect = self.detect_face(gray_frame)
        mask = self.get_roi_mask(gray_frame, face_rect)

        track_points = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)
        # Reshape into 2d (x,y) array
        #track_points = np.float32(track_points).reshape(-1, 2)
        return track_points


    @staticmethod
    def rect_to_bb(rect, up_scale=1.5):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()

        diff = int( h*up_scale - h )
        h = h*up_scale
        y = y - diff

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    @staticmethod
    def resize_face_rectange(x, y, w, h, r_w = 0.5, r_h=0.9):
        # Calulate new x and w values center
        new_w = w * r_w
        diff_w = w - new_w
        new_x = x + diff_w / 2 # Around center

        # Calculate new y and h values
        new_h = h * r_h
        diff_h = h - new_h
        new_y = y + diff_h / 2 
        new_y = new_y - new_h * 0.05

        return int(new_x), int(new_y), int(new_w), int(new_h)

    @staticmethod
    def remove_eyes_rectangle(x, y, w, h):
        # To do this we found that removing the subrectangle spanning 20% to 55% heightwise works wel
        new_y = y + y * 0.2
        new_h = h * (0.55 - 0.2)

        return int(x), int(new_y), int(w), int(new_h)



if __name__ == "__main__":

    face = FacePoints()

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        corners = face.get_points_pipeline(gray)

        if corners is not None:
            #corners = np.int0(corners)
            for i in corners:
                xc,yc = i.ravel()
                cv2.circle(vis,(xc,yc),3,255,-1)

            
        # Get rectangles
        x,y,w,h = face.face_rectange
        xx,yy,ww,hh = face.eyes_rectangle

        # Draw rectangle on face
        cv2.rectangle(vis, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.rectangle(vis, (xx,yy), (xx+ww,yy+hh),(0,0,255),2)

        # Show
        cv2.imshow('face track', vis)

        if cv2.waitKey(1) == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()
