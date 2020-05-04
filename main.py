from face import FacePoints
from tracking import TrackPoints
from signal_proc import get_diffs, filter_out, do_pca

import cv2
from tqdm import tqdm

import traceback
import logging
import json
from pprint import pprint

def dedect_bmp(video_src):
    result_dict = {
        'info': {
            'video_src': video_src,
            'fps': None,
            'frame_count': None,
            'duration': None,
        },
        'bpm' : None,
        'errors': None,
        'beat_on_frames': None,
    }


    capture = cv2.VideoCapture(video_src)

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps #secs

    result_dict['info']['fps'] = fps
    result_dict['info']['frame_count'] = frame_count
    result_dict['info']['duration'] = video_duration
     

    errors = []

    print('fps', fps, 'frame count', frame_count, 'duration', video_duration, 'seconds')

    gray_frames = [] #0 is newest -1 is oldest

    face = FacePoints()
    tracking = TrackPoints(max_trace_history=600)

    bpm = 0

    #while capture.isOpened():
    for frame_c in tqdm(range(frame_count)):
        # getting a frame
        ret, frame = capture.read()
        if not ret:
            error = 'Cant read frame %d, second: %f' % (frame_c, frame_c/fps)
            print(error)
            errors.append(error)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_frames.insert(0, gray)

        # Wait 5 frames before selecting points
        if frame_c >= 5:
            gray_frames.pop()
            
            try:
                tracking.track_points(gray_frames[1], gray_frames[0])
            except Exception as e:
                error = traceback.format_exc()
                print(error)
                errors.append(error)

                continue
    capture.release()


    # After watching video get signals

    try:
        diff = get_diffs(tracking.traces, fps)
        filtered_signals = filter_out(diff, fps)
        bpm, peaks = do_pca(filtered_signals, fps, show=False)
    except Exception as e:
        error = traceback.format_exc()
        print(error)
        errors.append(error)
        
    
    # Build json
    result_dict['errors'] = errors
    result_dict['bpm'] = bpm
    result_dict['beat_on_frames'] = peaks.tolist()

    pprint(result_dict)

    return json.dumps(result_dict)


if __name__ == "__main__":
    video_path = './data/face_videos/sitting.mkv'
    response = dedect_bmp(video_path)

    print(response)


