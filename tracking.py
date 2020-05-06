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


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


class TrackPoints:
    def __init__(self, face_dedector, max_trace_num=150, max_trace_history=60):

        self.traces = []
        self.max_trace_num = max_trace_num
        self.max_trace_history = max_trace_history
        self.track_started = False
        
        self.lastest_points = []

        self.face = face_dedector

        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def get_first_points(self, prev_frame, curr_frame):
        track_point_candidates = self.face.get_points_pipeline(prev_frame)

        if track_point_candidates is not None:

            if self.face.dedector_type == 'face_shape':
                initial_points = track_point_candidates
            else:
                initial_points = self.filter_unbacktrackable(prev_frame, curr_frame, track_point_candidates)

            # Add initial points
            for i, (x, y) in enumerate(np.float32(initial_points).reshape(-1, 2)):
                if i < self.max_trace_num:
                    self.traces.append([(x, y)])

            self.track_started = True
        
    def filter_unbacktrackable(self, prev_frame, curr_frame, track_point_candidates, ret_nextPts=False):
        if len(track_point_candidates) < 1:
            if not ret_nextPts:
                return []
            else:
                return [], []
        
        #Forward optical flow
        nextPts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, track_point_candidates, None, **self.lk_params)
        # Backward optical flow
        backNextPts, _st, _err = cv2.calcOpticalFlowPyrLK(curr_frame, prev_frame, nextPts, None, **self.lk_params) 
        
        # Find differance between 2 estimates
        # TODO: get distance
        dist = abs(track_point_candidates-backNextPts).reshape(-1, 2).max(-1)

        # Select backtraced points that are in 1 pixel dist
        bool_filter = dist < 1
        
        if not ret_nextPts:
            return track_point_candidates[bool_filter.flatten()]
        else:
            return nextPts, bool_filter

    def add_new_traces(self, prev_frame, curr_frame):
        track_point_candidates = self.face.get_points_pipeline(prev_frame)
        if track_point_candidates is not None:
            initial_points = self.filter_unbacktrackable(prev_frame, curr_frame, track_point_candidates)

            # Add initial points
            for x, y in np.float32(initial_points).reshape(-1, 2):
                if len(self.traces) < self.max_trace_num:
                    # Check if same as another point
                    if not [x,y] in self.lastest_points:
                        self.traces.append([(x, y)])        

    def filter_none_face(self, curr_frame):
        # Update face rectangle
        face_rect = self.face.detect_face(curr_frame)
        mask = self.face.get_roi_mask(curr_frame, face_rect)

        if face_rect is not None:
            new_lastest_points = []
            new_traces = []
            for i, (x,y) in enumerate(self.lastest_points):

                # If inside of face region add new list
                if self.face.point_in_rectangle(x,y, *self.face.face_rectange):
                    if not self.face.point_in_rectangle(x,y, *self.face.eyes_rectangle):
                        #print(i, 'in face')
                        new_lastest_points.append(self.lastest_points[i])
                        new_traces.append(self.traces[i])
                
            self.lastest_points = new_lastest_points
            self.traces = new_traces


    def track_points(self, prev_frame, curr_frame):
        if not self.track_started:
            self.get_first_points(prev_frame, curr_frame)
            if not self.track_started:
                return
            
        # Get previous frames from traces
        prevPts = np.float32([tr[-1] for tr in self.traces]).reshape(-1, 1, 2)
        nextPts, bool_filter = self.filter_unbacktrackable(prev_frame, curr_frame, prevPts, ret_nextPts=True)
        nextPts = nextPts.reshape(-1, 2)

        # Reset tracking
        if len(nextPts) < 1:
            self.track_started = False
            return

        # TODO: a hacky implementation
        if self.face.dedector_type == 'face_shape':
            
            # Get face points
            points = self.face.get_points_pipeline(curr_frame)
            points = np.array(points)

            # check if every point is at (0,0)
            if (points == 0).all():
                # Face is not dedected use lastest points
                return

            # Check untracable points and replace them with facepoints (int)
            idx = np.where(bool_filter == False)
            nextPts[idx] = points[idx]

            new_traces = []
            self.lastest_points = []
            # add from starting point
            for trace, (x,y) in zip(self.traces, nextPts):

                trace.append((x,y))
                self.lastest_points.append([x, y])

                # If trace history gets too big delete oldest element
                if len(trace) > self.max_trace_history:
                    del trace[0]

                new_traces.append(trace)
            
            self.traces = new_traces            
        else:

            new_traces = []
            self.lastest_points = []
            # add from starting point
            for trace, (x,y), good_flag in zip(self.traces, nextPts.reshape(-1, 2), bool_filter):
                # Delete unbacktrackable traces
                if not good_flag:
                    continue

                trace.append((x,y))
                self.lastest_points.append([x, y])

                # If trace history gets too big delete oldest element
                if len(trace) > self.max_trace_history:
                    del trace[0]

                new_traces.append(trace)
            
            self.traces = new_traces

            # Filter out points outside face mask
            #self.filter_none_face(curr_frame)

            # Add new traces if it shrink
            if len(self.traces) < self.max_trace_num:
                self.add_new_traces(prev_frame, curr_frame)


    def get_current_points(self):
        return np.int32([tr[-1] for tr in self.traces]).reshape(-1, 1, 2)


if __name__ == "__main__":

    

    capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture('./data/face_videos/standing.mkv')
    frame_c = 0
    gray_frames = [] #0 is newest -1 is oldest

    face = FacePoints(dedector_type='face_shape')
    tracking = TrackPoints(face_dedector=face)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))


    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        gray_frames.insert(0, gray)


        # Wait 10 frames before selecting points
        if frame_c >= 10:
            gray_frames.pop()

            tracking.track_points(gray_frames[1], gray_frames[0])
            nextPts = tracking.get_current_points()

            # Draw points
            for i, new in enumerate(nextPts):
                a,b = new.ravel()
                vis = cv2.circle(vis,(a,b),5,color[i%100].tolist(),-1)

            # Draw Tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracking.traces], False, (0, 255, 0))

            draw_str(vis, (20, 20), 'trace count: %d' % len(tracking.traces))
        # Show
        cv2.imshow('Track points', vis)

        if cv2.waitKey(1) == 27:
            break

        frame_c += 1

    capture.release()
    cv2.destroyAllWindows()