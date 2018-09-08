import math
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pyaudio 
import struct
import sys 
import audioop

def list_devices():
    """list out all the audio input devices"""
    player = pyaudio.PyAudio()
    i = 0 
    n = player.get_device_count()
    while i < n:
        dev = player.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(str(i) + '.' + dev['name'])
        i += 1

def calculate_energy(chunk):
    return float64(sum( [ abs(x)**2 for x in chunk ] ) / len(chunk))

def calculate_sfm(chunk):
    """Calculates the Spectral Flatness Measure of a signal
     The SFM is defined as the ratio of the geometrical mean by the
     arithmetical mean
    :param frame: frame of a discrete signal
    :return: the SFM of the frame
    """
    a = np.mean(chunk)
    g = gmean(chunk)
    if a == 0 or g/a <= 0:
        sfm = 0
    else:
        sfm = 10*np.log10(g/a)
    return sfm

def calculate_max_level(chunk):
    fmt = "%dH"%(len(chunk)/2)
    data_np = struct.unpack(fmt, chunk)
    data_np = np.array(data_np, dtype = 'h')

    data = np.fft.rfft(data_np)
    fft = np.abs(data)/len(data)
    max_freq_index = np.argmax(fft)

    freq = fft[max_freq_index]

    return freq

def extract_features(chunk):
    """ Given a signal, the number of frames, and the frame size, returns
     the energy, dominating frequency, and the sfm of all frames of the signal
    :param signal: A discrete signal
    :param num_frames: Number of frames of the signal
    :param frame_size: How many values are in a frame of the signal
    :param f_sampling: Sampling frequency
    :return: Returns 3 arrays of length 'num_frames' with the values of
             energy, dominating frequency, and sfm
    """
    num_frames = len(chunk)

    energy = np.array(np.zeros([num_frames]), dtype=int)
    energy2 = np.array(np.zeros([num_frames]), dtype=int)
    dominating_freq = np.array(np.zeros([num_frames]))
    sfm = np.array(np.zeros([num_frames]), dtype=int)

    # Calculating features (Energy, SFM, and most dominant frequency)
    for i in xrange(int(num_frames)):
        energy[i] = calculate_energy(chunk) #Energy 
        dominating_freq[i] = calculate_max_level(chunk)
        sfm[i] = calculate_sfm(chunk)

    return energy, dominating_freq, sfm

def remove_silence(signal, frame_size, speech):
    """ Gets a signal and remove its silence frames
    :param signal: A discrete signal
    :param frame_size: Number of samples in a frame
    :param speech: A bool array that has the info if a frame is silence or not
    :return: The resulting signal without silence frames
    """
    for i in xrange(len(speech)):
        if not speech[i]:
            signal[frame_size*i:frame_size*(i+1)] = 0

    result = signal[np.nonzero(signal)]

    return result


def audio_analysis():
    chunk = 2**11 
    samplerate = 16000

    device = 0

    # Setting the initial variables
    frame_size_time = 0.010
    frame_size_n = (samplerate * frame_size_time)
    num_frames = int(chunk)
    energy_prim_thresh = 40
    f_prim_thresh = 185
    sf_prim_thresh = 5
    player = pyaudio.PyAudio()

    info = player.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
        if player.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
                print("Input Device id ", i, " - ", player.get_device_info_by_host_api_device_index(0,i).get('name'))

        if player.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
                print("Output Device id ", i, " - ", player.get_device_info_by_host_api_device_index(0,i).get('name'))
    devinfo = player.get_device_info_by_index(0)
    print("Selected device is ",devinfo.get('name'))
    
    stream = player.open(format = pyaudio.paInt16, channels = devinfo['maxInputChannels'], rate = samplerate, frames_per_buffer = chunk, input_device_index = devinfo["index"], output = True)

    #print "Signal size: " + str(signal.size)
    #print "Freq Sampling: " + str(f_sampling) + " Hz"
    #print "Frame size: " + str(frame_size_n)
    #print "Number of frames: " + str(num_frames)

    print("starting analysis ... use Ctrl + C to stop")

    try:
        while True:
            data = stream.read(chunk, exception_on_overflow = False) 

            energy, dominating_freq, sfm = extract_features(data)

            min_energy = np.min(energy[0:30])
            min_sfm = np.min(sfm[0:30])

            thresh_energy = energy_prim_thresh * np.log10(min_energy)
            thresh_sfm = sf_prim_thresh

            speech = np.array(np.zeros([chunk]), dtype = bool)
            silence_count = 0

            # Deciding if a frame is a speech or silence
            for i in xrange(num_frames):
                counter = 0

                if energy[i] - min_energy >= thresh_energy:
                    counter += 1
                #if dominating_freq[i] - min_freq > thresh_freq:
                    counter += 1
                if sfm[i] - min_sfm >= thresh_sfm:
                    counter += 1

                # Not considering last frame
                # TODO: 0 padding in the last frame
                if counter > 0 and i != num_frames - 1:
                    speech[i] = True
                else:
                    speech[i] = False
                    silence_count += 1
                    min_energy = ((min_energy*silence_count)+energy[i])/(silence_count + 1)

             #print "Silence frames: " + str(silence_count) + " (before)"

            # Ignore silence run less than 10 successive frames
            # Ignore speech run less than 5 successive frames
            last = speech[0]
            sequence = 0
            start = 0
            for i in xrange(len(speech)):
                if last == speech[i]:
                    sequence += 1
                else:
                    if last is False and sequence < 10:
                        for j in xrange(start, i):
                            speech[j] = True
                    elif last is True and sequence < 5:
                        for j in xrange(start, i):
                            speech[j] = False
                    start = i
                    sequence = 0

            #print "Silence frames: " + str(num_frames - sum(speech)) + " (after)"

            result = remove_silence(signal, frame_size_n, speech)
            result2 = np.array(result, np.int16)

            #print "Result signal size: " + str(len(result2))

            return result2
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping")
        stream.close()
        player.terminate()
        print("Done")

if __name__ == '__main__':
    list_devices()
    audio_analysis()