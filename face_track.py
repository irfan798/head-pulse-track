import time

import numpy as np
import cv2
from imutils import face_utils
import dlib

from scipy import interpolate, signal, fftpack, optimize
from sklearn.decomposition import PCA
import pylab as plt
#plt.interactive(True)


# hard coded parameters (beark... some of them need to be passed into init)
# parameters of the Lucas-Kanade optical flow algorithm
lk_params = dict( winSize  = (35, 35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# parameters of the feature tracking algorithm
feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.01,  # decrease sensitivity
                      minDistance = 3,
                      blockSize = 7 )

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predict_dat = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predict_dat)


# Haar
faceCascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

crop_height = 0.45
crop_width = .5


def crop_top_face(face):
    fh = int(face[3] * crop_height)
    fw = int(face[2] * crop_width)
    # remove the forehead and crop the face tracking rectangle
    return face[0]+fw/2, face[1]+fh, face[2]-fw, face[3]-fh

def crop_bottom_face(face):
    fh = int(face[3] * (crop_height-.25))
    fw = int(face[2] * crop_width)
    return face[0]+fw/2, face[1], face[2]-fw, fh

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


def detect_faces(frame):

    #dlib
    #face_rects = detector(frame, 0)
    #faces = [rect_to_bb(face_rect) for face_rect in face_rects ]
    #(x, y, w, h) = rect_to_bb(face_rect)

    faces = faceCascade.detectMultiScale(frame)
    return faces


def resize_rectange(x, y, w, h, r_w = 0.5, r_h=0.9):
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

def remove_eyes_rectangle(x, y, w, h):
    # To do this we found that removing the subrectangle spanning 20% to 55% heightwise works wel
    new_y = y + y * 0.2
    new_h = h * (0.55 - 0.2)

    return int(x), int(new_y), int(w), int(new_h)



def create_mask(img, face_rect, eyes_rect):
    x,y,w,h = face_rect
    xx,yy,ww,hh = eyes_rect

    mask = np.zeros(img.shape, np.uint8)
    mask[y:y+h,x:x+w] = 255 # Add face
    mask[yy:yy+hh,xx:xx+ww] = 0 # Remove eyes 

    return mask


capture = cv2.VideoCapture(0)

while capture.isOpened():
    # getting a frame
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    face_rects = detect_faces(gray)
    # Get areas of faces
    face_sizes = [h*w for x, y, w, h in face_rects]

    if len(face_rects) > 0:
        # Then get biggest face
        face_idx = np.argmax(face_sizes)
        face_rect = face_rects[face_idx]

        # get face coordinates
        (x, y, w, h) = face_rect
        #(x, y, w, h)  = (int(x), int(y), int(w), int(h) )

        # Resize rectange to %55 widthwise %90 Height wise
        (x, y, w, h) = resize_rectange(x, y, w, h)
        (xx, yy, ww, hh) = remove_eyes_rectangle(x, y, w, h)

        #Crop face on gray

        # Create mask of face rectangle
        mask = create_mask(gray, (x, y, w, h), (xx, yy, ww, hh))

        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        if corners is not None:
            #corners = np.int0(corners)
            for i in corners:
                xc,yc = i.ravel()
                cv2.circle(vis,(xc,yc),3,255,-1)

        
        
        # Draw rectangle on face
        cv2.rectangle(vis, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.rectangle(vis, (xx,yy), (xx+ww,yy+hh),(0,0,255),2)

    
    # Show
    cv2.imshow('face track', vis)

    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()
