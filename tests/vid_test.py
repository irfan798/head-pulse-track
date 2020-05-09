import numpy as np
import cv2

cap = cv2.VideoCapture('./data/face_videos/sitting.mkv')

fps = int(cap.get(cv2.CAP_PROP_FPS))

print('fps', fps)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()