# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from pprint import pprint
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)

    
    left_eye = list(range(36,42)) # 36 to 41
    right_eye = list(range(42,48)) # 42 to 47
    mouth = list(range(48, 69))  #48 
    remove = left_eye + right_eye + mouth
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for j, (x, y) in enumerate(shape):
            if j in remove:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            else:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()