[![Documentation Status](https://readthedocs.org/projects/eoglearn/badge/?version=latest)](https://eoglearn.readthedocs.io/en/latest/?badge=latest)

This package contains tools for denoising EOG artifact using simultaneously collected M/EEG & eye-tracking data. This project is currently in the proof-of-concept stage and we consider this software as pre-alpha, meaning that it is not ready to be used by end users. However, we include documentation (API, tutorials), and helper functions for loading open-access eeg-eyetracking data, as to make contributing to the development of this project feasible for anyone that is interested. See the road map below


Problem Statement
=================

MEG and EEG data contain a mixture of neural signal and non-neural artifact (for example muscle movment, heartbeat, eye blinks, and eye saccades). Eye blinks and eye saccades (ocular artifacts) are always present in M/EEG and have large amplitude as compared to neural signal. M/EEG devices record electrical signals in the brain - interestingly, an electrical dipole exists in the human eye, and thus this is inadverdently picked up by the M/EEG when we move our eyes etc. Currently, most MEEG analyses use ICA (blind source separation) to "subtract" the ocular artifact from the MEEG signal. This has the benefit of retaining the full M/EEG time-course (ie. before the days of ICA, scientists would cut out entire segments of M/EEG with ocular artifact in them, resulting in a loss of data). However, ICA is an unsupervised method, it attempts to identify "patterns" in the data. Usually, it will identify the ocular artifact patttern in M/EEG, however this is not a perfect approach, and it is not known how much neural signal is lost when subtracting out the ocular ICA components. Further, ICA assumes that all sources in the data are independent. This is not always the case for M/EEG data, raising the question of whether the approach is valid for M/EEG.

Aims
====

Integrated M/EEG-Eyetracking recordings are becoming increasingly more common. For these recordings, we know exactly what the eye is doing at any point in time, and thus, we may not need to rely on unsupervised methods for estimating the ocular signal in M/EEG. Instead, we aim to develop and test a self-supervised approach to isolating the EOG artifact in M/EEG data using intregrated M/EEG-Eyetracking recordings and a Machine Learning framework

Model
=====

As of this writing, this is the set-up of the model: We use a Long-Short-Term-Memory framework because it is capable of identifying temporal dependencies, and can be used as a "many-to-many" model. I.e., we can use the eyetracking signals to predict the EOG artifact in the EEG signals. Basically, the model currently attempts to produce the part of the EEG signal that is learnable from the eye-tracking signals. Then we should be able to subtract this signal from the actual M/EEG data, to remove the EOG artifact<sup>**</sup>.

The image below assumes 5 eye-tracking channels as input and 129 EEG channels as output, which is the case for one of the current test files. However, in practice this model should be able to take  as little as 3 eye-tracking channels (x-coordinate position, y-coordinate position, and pupil size), and a variable numbe rof output M/EEG signals, depending on which manufacture was used. Currently, it is assume that this model will take an integrated MEEG-eyetracking recording as input. I.e., Before this model can be used with MEEG data without corresponding eye-tracking data, it will likely need to be trained on multiple datasets from differing MEEG systems, so that the model will generalize well to unseen data.

![eeget_LSTM](https://github.com/scott-huberty/eog-learn/assets/52462026/838b4559-9c95-4cd9-b452-ba95d2ff2d42)



Road Map
========

- [ ] **Proof-of-concept**: Demonstrate the ability to use ML with simultaneously collected M/EEG-Eyetracking data to denoise EOG artifact from the M/EEG data, using one or two files.
- [ ]  **Train Model with open-access data**: after a proof-of-concept has been demonstrated, it should be applied to 1 or more open-access datasets for hyper-parameter training and model compiling. The [EEGEyeNet dataset](https://osf.io/ktv7m/) is a strong candidate for this step.
- [ ]  **Compare with current state-of-the-art tools**. Compare M/EEG data cleaned with this tool to the same data cleaned with ICA and the ICLabel classifier, which uses Deep 
       Learning to classify ICA components as EOG, Muscle, Brain, etc. Metrics for comparing the output of these two approahces TBD.



------------------------------------------
<sup>**</sup>A possible "*gotcha*" with this model, is that it may learn _any_ M/EEG signal that is temporally correlated with the eye-tracking signal. This may pose a problem in the event that there is a neural response that is highly correlated with eye-movement, for example take an anti-saccade task where there may be a visual neural response the moment the participant finishes a saccade to fix their gaze to the target. The model may learn to associate this ERP with the eye-tracking data, and incorrectly consider it to be EOG. In this case, it's likely that that the signal-to-noise ratio of the evoked data would be smaller _after_ applying this model than before (because the ERP is being regressed out). As part of the proof-of-concept of this model, it should be demonstrated that we can isolate EOG artifact without inadverdently throwing out neural data.

