# Detecting Pulse from Head Motions in Video
This is the python implementation of this [paper](https://people.csail.mit.edu/mrub/vidmag/papers/Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf)

## Explanation
This is abstract directly quated from paper
> We extract heart rate and beat lengths from videos by
measuring subtle head motion caused by the Newtonian
reaction to the influx of blood at each beat. Our method
tracks features on the head and performs principal compo-
nent analysis (PCA) to decompose their trajectories into a
set of component motions. It then chooses the component
that best corresponds to heartbeats based on its temporal
frequency spectrum. Finally, we analyze the motion pro-
jected to this component and identify peaks of the trajecto-
ries, which correspond to heartbeats. When evaluated on
18 subjects, our approach reported heart rates nearly iden-
tical to an electrocardiogram device. Additionally we were
able to capture clinically relevant information about heart
rate variability.

![project overview](data/overview.png)


## Warning
This implementation is made in 3 days span for a hackathon so there are still a lot of bugs and easy fix

All push requests and questions are welcome

# Usage

From the project root folder
```bash
python modules/signal_processing.py
```

If you want to test other modules use:
```
python modules/face.py
python modules/tracking.py
```

# Project Structure

Project consists of 3 modules for now:

* Face - FacePoints
* tracking - TrackPoints
* signal_processing -

```bash
├── main.py
├── modules
│   ├── face.py
│   ├── signal_proc.py
│   └── tracking.py
├── papers
│   ├── Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf
│   ├── electronics-08-00663.pdf
├── README.md
├── requirements.txt
├── tests
│   ├── common.py
│   ├── dlib_dedect.py
│   ├── fft.ipynb
│   ├── lk_track.py
│   ├── signal_pross.ipynb
│   └── vid_test.py
├── data
│   ├── haarcascade
│   │   ├── ***
│   └── shape_predictor_68_face_landmarks.dat


```

## __FacePoints__
This modules is for dedecting face and getting initial points on face you can select one of the followings with *dedector_type* argument

  1.  __haar__ 
   * OpenCV's Haar face detection, gets face rectangle, rescales and removes eyes. Then finds most optically trackable points in that area
   * Fastest one
  2.  __dlib__
   * Same as Haar but this time uses Dlib's face dedection
   * More accurate
  3.  __face_shape__
   * After detecting face rectangle with dlib finds 68 face landmarks on face. Then removes eyes and mount landmark points to getting more stabilized data

## __TrackPoints__
This module is responsible for tracking detected points throughout video
Takes
`face_dedector, max_trace_num=150, max_trace_history=60` as input

__Face_dedector__ is __FacePoints__ class

If `Haar` or `Dlib` is used on face detection then tracker checks every point with backtracking, if a point is not consistent with forward and backtracking then trace is marked as faulty and deleted.

Every time trace non faulty traces are counted if number is not equal to `max_trace_num` then new trackable points are requested from `face_detector`

If `face_shape` is used on face detection then initial landmark points are tracked again with optical flow, if a point on face landmarks marked as faulty by backtracker then face landmarks are requested from `face_detector` but only faulty points are replaced 

Because face detection only gives points as integer but opticalflow tracks on subpixels and returns float, If instead of tracking initial points we request landmark everyframe we would miss a lot of motion happening on subpixels.


## __signal_processing__
This module is responsible for detecting hearth beat from tracked signals.

First we only get y components from tracked traces and then filter them with 5th order butterworth bandpass filter to get only frequencies between [0.75Hz, 3Hz] -> (45bpm, 180bpm)

Third if `Haar` or `dlib` used as initial points, does PCA to get most meaningful points to track with dimesion reduction to 5 components.

With those 5 components we check most dominant frequency in each of them using with FFT and get percentage of total
spectral power accounted for by the frequency with maxi-
mal power

Only most periodic one is selected in the and

### __IF face_shape is used__
Instead of selecting indivitual points with PCA, we get mean of every traces pixel movement, as a result we got 1 signal then get the dominant frequency in that signal with FFT and return BPM accordingly.

## TODO:
- [ ] Make main with source selection and argument parsing
- [ ] Make signal_processing a module
- [ ] Make plotting faster
- [ ] For much better signal processing, add timestamp to every points tracked location in __TrackPoints__
- [ ]  For live feed first get real fps with calculating time between every frame, then use it as __sampling rate__