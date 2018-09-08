#test_gender.py
from __future__ import division
import os
from queue import Queue
import pickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
from scipy import signal
from pysndfx import AudioEffectsChain
import librosa
from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
import sys

from parabolic import parabolic

warnings.filterwarnings("ignore")


def get_MFCC(sr,audio):

    features = mfcc.mfcc(audio,sr)

    #############################
    #                           #
    #      Noise Removal        #
    #                           #
    #############################

    features = mfcc.logfbank(audio) #computes the filterbank energy from an audio signal
    features = mfcc.lifter(features) #increases magnitude of high frequency DCT coefficients 

    sum_of_squares = []
    index = -1

    for r in features: 
        """
        Since signals can be either positive or negative, taking n**2 allows us to compare the magnitudes 
        """
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares)) 
    hz = mfcc.mel2hz(features[strongest_frame]) #converts the strongest frame's mfcc to hertz

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=20.0, slope=0.5) #creates an audio booster that removes low hz
    y_speech_boosted = speech_booster(audio) #apply booster to original audio

    #############################
    #                           #
    #  FINAL MFCC CALCULATION   #
    #                           #
    #############################

    features = mfcc.mfcc(y_speech_boosted, sr, 0.025, 0.01, 16, nfilt=40, nfft=512, appendEnergy = False, winfunc=np.hamming)


    features = preprocessing.scale(features) #scaling to ensure that all values are within 0 and 1

    return features

def freq_from_autocorr(sr, audio):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(audio, audio[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return sr / px


def freq(audio, sr):
    i = freq_from_autocorr(audio, sr)

    if i >= 85 and i <= 300:
        return i

#path to testing data
sourcepath = "/Users/huangel/Desktop/MIT 2017-2018/2018 Summer/Via Technologies Internship/VAD/test_data"   

#path to saved models    
modelpath  = "/Users/huangel/Desktop/MIT 2017-2018/2018 Summer/Via Technologies Internship/VAD"  

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
files     = [os.path.join(sourcepath,f) for f in os.listdir(sourcepath) if f.endswith(".wav")] #load all the wave files

def get_score(features):
    global models
    scores     = None
    log_likelihood = np.zeros(len(models)) 
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model (ie/ female and male) one by one
        scores = np.array(gmm.score(features)) #comparing
        log_likelihood[i] = scores.sum() #since gmm.score returns array instead of float
    s_sdc = log_likelihood[1] - log_likelihood[0]
    print(s_sdc)
    if s_sdc > 0:
        return 'noise'
    else:
        return 'speech'

import pyaudio
q = Queue()
player = pyaudio.PyAudio()
stream = player.open(format = pyaudio.paInt16, channels = 0, rate = 16000, input = True, frames_per_buffer = 2**11, input_device_index = 1)

try:
    while run:
        data = stream.read(chunk)
        spectrum = get_MFCC(16000, data)
        preds = detect_triggerword_spectrum(spectrum)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        if new_trigger:
            sys.stdout.write('1')
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False
        
stream.stop_stream()
stream.close()
