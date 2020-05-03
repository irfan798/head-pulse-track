import time

import numpy as np
import cv2
from imutils import face_utils
import dlib
from face import FacePoints

## Select REGION


## Start Tracking points


## Get longitunial trajectories

## Temporal filtering

## PCA 
### Get most periodic one

## Peak dedection

## Hearthbeat

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


if __name__ == "__main__":

    face = FacePoints()

    capture = cv2.VideoCapture(0)
    frame_c = 0
    gray_frames = [] #0 is newest -1 is oldest
    frames = []

    old_points = []

    # Create some random colors
    color = np.random.randint(0,255,(100,3))


    track_point_candidates = []
    ptsHistory = []
    traces = []
    max_trace_history = 60
    track_started = False

    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        gray_frames.insert(0, gray)

        mask = np.zeros_like(frame)

    

        # Wait 10 frames before selecting points
        if frame_c >= 10:
            gray_frames.pop()

            # Get better points

            if not track_started:
                # Get prev frame candidates
                track_point_candidates = face.get_points_pipeline(gray_frames[1])

                old_points = track_point_candidates

                #track_started = True

            if old_points is not None:
            
                #Forward calculate optical flow
                nextPts, st, err = cv2.calcOpticalFlowPyrLK(gray_frames[1], gray_frames[0], old_points, None, **lk_params)
                # Backward optical flow
                #backNextPts, _st, _err = cv2.calcOpticalFlowPyrLK(gray_frames[0], gray_frames[1], nextPts, None, **lk_params) 
                
                # Find differance between 2 estimates
                #d = abs(nextPts-backNextPts).reshape(-1, 2).max(-1)

                # Get points where distance is smaller then 1 px
                #goodPts = nextPts[d < 1]

                # Select good points
                bool_filter = st == 1
                bool_filter = bool_filter.flatten()
                good_new = nextPts[bool_filter]
                good_old = old_points[bool_filter]

                # Add initial points
                if not track_started:
                    for x, y in np.float32(good_old).reshape(-1, 2):
                        traces.append([(x, y)])

                    track_started = True
                
                new_traces = []
                # add from starting point
                for trace, (x,y) in zip(traces, good_new.reshape(-1, 2)):
                    trace.append((x,y))

                    # If trace history gets too big delete oldest element
                    if len(trace) > max_trace_history:
                        del trace[0]
                    # Why
                    new_traces.append(trace)
                
                #Why
                traces = new_traces


                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new, good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()

                    vis = cv2.circle(vis,(a,b),5,color[i%100].tolist(),-1)

                cv2.polylines(vis, [np.int32(tr) for tr in traces], False, (0, 255, 0))

                old_points = good_new

        # Show
        cv2.imshow('Track points', vis)

        if cv2.waitKey(1) == 27:
            break

        frame_c += 1

    capture.release()
    cv2.destroyAllWindows()