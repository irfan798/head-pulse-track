import numpy as np
import cv2
import argparse

from modules.face import FacePoints
from modules.tracking import TrackPoints
from modules.signal_processing import SignalProcess


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera index')
    parser.add_argument('-f', '--file', type=str, default=-1, help='Video path')
    parser.add_argument('-g', '--graph', action='store_true', help="Show graph")
    parser.add_argument('--face_detector', type=str, default='haar',
                        help='Face detector algorithm to be used ["haar","dlib","face_shape"]')
    parser.add_argument('--fps', type=int, default=-1, help='Force fps instead of autodetect')
    parser.add_argument('--trace_len', type=int, default=180,
                        help='Maximum trace length to be saved and used in calculating frequency')
    parser.add_argument('--trace_num', type=int, default=60,
                        help='Maximum points to be tracked, only effective if "haar" or "dlib" is used')
    args = parser.parse_args()

    # Use file instead of camera
    if not args.file == -1:
        capture = cv2.VideoCapture(args.file)
    else:
        capture = cv2.VideoCapture(args.camera)

    # Autodetect fps
    if args.fps == -1:
        fps = int(capture.get(cv2.CAP_PROP_FPS))
    else:
        fps = args.fps

    gray_frames = []  # 0 is newest -1 is oldest
    frame_c = 0

    face = FacePoints(dedector_type=args.face_detector)
    tracking = TrackPoints(face_dedector=face, max_trace_history=args.trace_len, max_trace_num=args.trace_num)
    signal = SignalProcess(tracking, fps, draw=args.graph)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        gray_frames.insert(0, gray)

        # Wait 3 frames before selecting points
        if frame_c >= 3:
            # Keep most recent 3 gray frames
            gray_frames.pop()

            # Track Face points
            tracking.track_points(gray_frames[1], gray_frames[0])

            # Get longest trace
            longest_trace = max( [len(trace) for trace in tracking.traces] )

            # Draw points
            pts = tracking.get_current_points()
            for i, new in enumerate(pts):
                a,b = new.ravel()
                vis = cv2.circle(vis,(a,b),5,color[i%100].tolist(),-1)

            # Draw Tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracking.traces], False, (0, 255, 0))

            draw_str(vis, (20, 100), 'trace lenght: %d' % longest_trace)
            draw_str(vis, (20, 60), signal.graph_message)
            draw_str(vis, (20, 20), 'bpm: %d' % signal.mean_bpm)

            # Only try to find BPM if trace len is longer then 3 seconds
            if longest_trace > 3*fps:
                signal.find_bpm()

        # Show
        cv2.imshow('Signal Process', vis)

        # Break with esc key
        if cv2.waitKey(1) == 27:
            break

        frame_c += 1

    capture.release()
    cv2.destroyAllWindows()