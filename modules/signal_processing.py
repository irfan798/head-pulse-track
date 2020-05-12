import numpy as np
import cv2

import matplotlib.pyplot as plt

from scipy import interpolate, signal, optimize
from scipy.fftpack import fft, ifft, fftfreq, fftshift

from sklearn.decomposition import PCA


class SignalProcess:

    def __init__(self, signal_source, fs=30, draw=False):

        self.signal_source = signal_source
        self.fs = fs

        self.bpm_list = []
        self.mean_bpm = 0

        self.draw = draw

        # Graphs
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.graph_message = ''

        if self.draw:
            self.fig.show()



    def get_diffs(self, traces, fps):
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

    def get_y(self, traces):
        """ Get Y coordinates of given traces """
        # Filter traces get 4sec or long nyquist freq for 0.5 hz -> 1 hZ
        traces = [trace for trace in traces if len(trace) > 2*self.fs]
        trace_max_len = max( [len(trace) for trace in traces] )

        #trace_max_len = 300
        #TODO: This is quickfix
        traces = [trace for trace in traces if len(trace) == trace_max_len]

        # Calculate y movement of each
        ys = []
        #displacements = np.array([[]])
        for trace in traces:
            trace = np.array(trace)[:, 1]

            ys.append(trace)
        return np.stack(ys, axis=0)

    def filter_signal(self, signal_data, fs=30, low_c=0.75, high_c=2.0):
        """
        This function bandpass filters given signal
        :param signal_data: Input Signal
        :param fs: Sampling Frequency
        :param low_c: Low Cutoff Frequency
        :param high_c: High Cutoff Frequency
        :return: Filtered signal
        """

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

    def filter_out(self, signals, low_c=0.5, high_c=2.0):
        """
        Filter multiple signals then return 2D array
        :param signals: Input signals
        :param low_c: Low Cutoff Frequency
        :param high_c: High Cutoff Frequency
        :return: numpy stacked filtered signals
        """
        filtered_signals = []

        for signal_data in signals:
            filter_out = self.filter_signal(signal_data, fs=self.fs, low_c=low_c, high_c=high_c)
            filtered_signals.append(filter_out)

        if len(filtered_signals) > 0:
            filtered_signals = np.stack(filtered_signals, axis=0)

        return filtered_signals[:-self.fs]

    def get_mean(self, filtered_signals, show=True):
        """ Reduces dimension of traces to 1 by getting mean vertically"""
        if len(filtered_signals) < 5:
            return 0

        mean_signal = np.mean(filtered_signals, axis=0, dtype=np.float64)
        maxFreq, percentage = self.get_dominant_frequency(mean_signal, fs=self.fs, draw=False)

        bpm = maxFreq * 60

        if show:
            self.ax1.cla()
            self.ax2.cla()
            self.get_dominant_frequency(mean_signal, fs=self.fs, draw=True)
            self.fig.canvas.draw()

        return bpm


    def get_dominant_frequency(self, signal_data, fs=30, draw=False):
        """
        Finds the dominant frequency with FFT
        :param signal_data: Filtered Signal Data
        :param fs: Sampling Frequency
        :param draw: Should draw with MatplotLib
        :return: maxFreq, percentage Dominant frequency and how strong it is in PowerSpectrum
        """
        # number of signal points
        N = len(signal_data)
        # sample spacing
        T = 1.0 / fs

        # Get fft
        spectrum = np.abs(fft(signal_data))
        spectrum *= spectrum
        xf = fftfreq(N, T)

        # Get maximum ffts index from second half
        #maxInd = np.argmax(spectrum[:int(len(spectrum)/2)+1])
        maxInd = np.argmax(spectrum)
        maxFreqPow = spectrum[maxInd]
        maxFreq = np.abs(xf[maxInd])

        total_power = np.sum(spectrum)
        # Get max frequencies power percentage in total power
        percentage = maxFreqPow / total_power

        if draw:
            t = np.linspace(0.0, T*N, N)

            self.ax1.set_title('Signal data')
            self.ax1.plot(t, signal_data)
            #self.ax1.plot(peaks/fps, signal_data[peaks], "x")
            #self.ax1.plot(np.zeros_like(t/fps), "--", color="gray")
            self.ax1.set(xlabel='Time', ylabel='Pixel movement')
            self.ax1.grid()

            self.ax2.plot(xf, 1.0/N * spectrum)
            self.ax2.set_title('FFT')
            self.ax2.axvline(maxFreq, color='red')
            self.ax2.grid()
            self.ax2.set(xlabel='Freq', ylabel='')

            #print("Max power Freq {} % {} BPM:{}".format(maxFreq, percentage, bpm))

        return maxFreq, percentage


    def do_pca(self, filtered_signals, fps, show=True):
        """
         Reduces signals into 5 channels by PCA then finds dominant frequency on each of them and
         selects the most dominant one based on how much power it has on total power
         """
        if len(filtered_signals) < 5:
            return 0

        pca = PCA(n_components=5)
        pca_result = pca.fit_transform(filtered_signals.T).T

        max_ratios = []
        max_freqs = []
        for i, signal_data in enumerate(pca_result):
            maxFreq, percentage = self.get_dominant_frequency(signal_data, fs=fps, draw=False)
            max_ratios.append(percentage)
            max_freqs.append(maxFreq)

        # Find most dominant out of pcas
        idx = np.argmax(max_ratios)
        last_pca = pca_result[idx]

        self.graph_message = "Selected PCA:{}".format(idx)

        bpm = max_freqs[idx]*60

        if show:
            self.ax1.cla()
            self.ax2.cla()
            self.get_dominant_frequency(last_pca, fs=fps, draw=True)
            self.fig.canvas.draw()
        return bpm

    def find_bpm(self, bpm_list_len=10, low_c=0.5, high_c=3.0):

        bpm = 0

        traces = self.get_y(self.signal_source.traces)
        filtered_signals = self.filter_out(traces, low_c=low_c, high_c=high_c)

        # If finding bpm from face shape
        if self.signal_source.face.dedector_type == 'face_shape':
            bpm = self.get_mean(filtered_signals, self.draw)
        else:
            bpm = self.do_pca(filtered_signals, self.fs, show=self.draw)

        self.bpm_list.insert(0, bpm)

        if len(self.bpm_list) > bpm_list_len:
            self.bpm_list.pop()

        self.mean_bpm = sum(self.bpm_list) / len(self.bpm_list)

        return bpm
