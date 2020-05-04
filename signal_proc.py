import time

import numpy as np
import cv2
from imutils import face_utils
import dlib
from face import FacePoints
from tracking import TrackPoints

import matplotlib.pyplot as plt


from scipy import interpolate, signal, optimize
from scipy.fftpack import fft, ifft, fftfreq, fftshift

from sklearn.decomposition import PCA

from scipy.signal import find_peaks


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



def get_diffs(traces, fps):
    # Filter traces get 4sec or long nyquist freq for 0.5 hz -> 1 hZ
    traces = [trace for trace in traces if len(trace) > 2*fps]
    trace_max_len = max( [len(trace) for trace in traces] )

    #trace_max_len = 300
    #TODO: This is quickfix
    traces = [trace for trace in traces if len(trace) == trace_max_len]   

    # Calculate y movement of each
    displacements = []
    #displacements = np.array([[]])
    for trace in traces:
        trace = np.array(trace)

        y_pts = trace[:, 1]
        # Pad array to standart lenght
        len_diff = trace_max_len-len(y_pts)
        if len_diff > 0:
            pass
            print('Padded', len_diff)    
        y_pts = np.pad(y_pts, (len_diff, 0), 'edge')
        
        displace = np.diff(y_pts) # y coordinates
        displacements.append(displace)

    if len(displacements) > 0:
        displacements = np.stack(displacements, axis=0)

    return displacements


# Filter Signal
def filter_signal(signal_data, fs=30, low_c=0.75, high_c=2.0):

    #fs = 30 # Fps
    # number of signal points
    N = len(signal_data)
    # sample spacing
    T = 1.0 / fs

    #Draw signal
    #t = np.arange(len(displace))/fps
    t = np.linspace(0.0, T*N, N)

    # Filter signal
    fc = np.array([low_c, high_c])  # Cut-off frequency of the filter
    # 0.75 hz - 2 hz => 45bpm - 120bpm

    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'bandpass')

    filter_output = signal.filtfilt(b, a, signal_data)
    
    return filter_output


def filter_out(displacements, fps):
    filtered_signals = []

    for signal_data in displacements:
        filter_out = filter_signal(signal_data, fs=fps)
        filtered_signals.append(filter_out)

    if len(filtered_signals) > 0:
        filtered_signals = np.stack(filtered_signals, axis=0)

    return filtered_signals

def do_pca(filtered_signals, fps, show=True):
    if len(filtered_signals) < 5:
        return 0
    
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(filtered_signals.T).T

    x = pca_result[1]
    peaks, _ = find_peaks(x, height=0)

	#i=t[len(t)-1]
	#ax.set_xlim(left=max(0, i-15), right=i+2)
	
    if show:
        ax.cla()
        ax.plot(x)
        ax.plot(peaks, x[peaks], "x")
        ax.plot(np.zeros_like(x), "--", color="gray")
        #ax.set_ylim(bottom=min(x),top=max(x))
        fig.canvas.draw()
        #plt.show()

    total_secs = len(x)/fps
    total_beats = len(peaks)
    bps = total_beats / total_secs

    #print(bps*60)
    return bps*60, peaks



if __name__ == "__main__":

    #capture = cv2.VideoCapture('./data/face_videos/sitting2.avi')
    capture = cv2.VideoCapture(0)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    print('fps', fps)

    gray_frames = [] #0 is newest -1 is oldest
    bpm_list = [] #0 is newest -1 is oldest
    frame_c = 0

    face = FacePoints()
    tracking = TrackPoints(max_trace_history=300, max_trace_num=60)

    # Create some random colors
    color = np.random.randint(0,255,(100,3))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()


    while capture.isOpened():
        # getting a frame
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        gray_frames.insert(0, gray)


        # Wait 10 frames before selecting points
        if frame_c >= 3:
            gray_frames.pop()

            tracking.track_points(gray_frames[1], gray_frames[0])
            nextPts = tracking.get_current_points()

            # Draw points
            for i, new in enumerate(nextPts):
                a,b = new.ravel()
                vis = cv2.circle(vis,(a,b),5,color[i%100].tolist(),-1)

            # Draw Tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracking.traces], False, (0, 255, 0))



            # Calculate distance travalled by tracks
            trace_max_len = max( [len(trace) for trace in tracking.traces] )

            draw_str(vis, (20, 100), 'trace lenght: %d' % trace_max_len)


            if trace_max_len > 3*fps:
                diff = get_diffs(tracking.traces, fps)
                filtered_signals = filter_out(diff, fps)
                bpm = do_pca(filtered_signals, fps)

                bpm_list.insert(0, bpm)

                if len(bpm_list) > 10:
                    bpm_list.pop()

                    mean_bpm = sum(bpm_list) / len(bpm_list) 

                    draw_str(vis, (20, 20), 'bpm: %d' % mean_bpm)


        # Show
        cv2.imshow('Signal Process', vis)

        if cv2.waitKey( int(1) ) == 27:
            break

        frame_c += 1

    capture.release()
    cv2.destroyAllWindows()