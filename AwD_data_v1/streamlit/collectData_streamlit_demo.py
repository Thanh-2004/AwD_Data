import numpy as np
import serial
import random
import pandas as pd
import scipy as sp
import queue
import threading
from extract_feature import FeatureExtract
import pickle
import time 

# def FeatureExtract(data):
#     f, t, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
#     delta = np.array([], dtype=float)
#     theta = np.array([], dtype=float)
#     alpha = np.array([], dtype=float)
#     beta = np.array([], dtype=float)
#     for i in range(0, int(t[-1])):
#         indices = np.where((f >= 0.5) & (f <= 4))[0]
#         delta = np.append(delta, np.sum(np.abs(Zxx[indices, i])))

#         indices = np.where((f >= 4) & (f <= 8))[0]
#         theta = np.append(theta, np.sum(np.abs(Zxx[indices, i])))

#         indices = np.where((f >= 8) & (f <= 13))[0]
#         alpha = np.append(alpha, np.sum(np.abs(Zxx[indices, i])))

#         indices = np.where((f >= 13) & (f <= 30))[0]
#         beta = np.append(beta, np.sum(np.abs(Zxx[indices, i])))

#     abr = alpha / beta
#     tbr = theta / beta
#     dbr = delta / beta
#     tar = theta / alpha
#     dar = delta / alpha
#     dtabr = (alpha + beta) / (delta + theta)

#     diction = {"delta": delta,
#                "theta": theta,
#                "alpha": alpha,
#                "beta": beta,
#                "abr": abr,
#                "tbr": tbr,
#                "dbr": dbr,
#                "tar": tar,
#                "dar": dar,
#                "dtabr": dtabr
#                }
#     return diction

### Random
def get_AS(model, feature):
    # awake_score = random.randint(0,100)
    # status = random.choice(["Awake", "Fatigue"])
    # probabilities = list(np.random.randint(0, 101, size=5))
    probabilities = model.predict_proba(feature)[0]
    print(probabilities)
    out = [0, 25, 50, 75, 100]
    awake_score = int(np.dot(probabilities, out))

    if awake_score > 50:
        status = "Awake"
    else:
        status = "Fatigue"

    return awake_score, status, probabilities

def get_signal():
    signal = random.randint(0,100)
    return signal


def collectData(path, time_rec, port, q: queue):
    # if serial.Serial:
    #     serial.Serial().close()
    # # Open the serial port
    # s = serial.Serial(port, baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
    path = "/Users/nguyentrithanh/Documents/Lab/EEG_AwakeDrive/AD_Code/AwD_Data/AwD_data_v1/test_data/BachCalm.txt"
    file = open(path, "r")

    x = 0  # iterator of sample
    y = np.array([], dtype=int)  # value
    k = 15  # window
    sample_rate = 512
    window_size = k * sample_rate

    model_path = 'gradient.pkl'

    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)


    print("START!")
    while x < (time_rec * 512):
        if x % 512 == 0:
            second = x//512
            print(second)
            time.sleep(0.6)

        x += 1
        data = file.readline()

        y = np.append(y, int(data))

        if x >= window_size:
            if x % (1 * sample_rate) == 0:
                sliding_window_start = x - window_size
                sliding_window_end = x
                slide = np.array(y[sliding_window_start:sliding_window_end]) 
                feature_window = FeatureExtract(slide)

                feature_slide = [np.array(list(feature_window.values()))]

                awake_score, status, probabilities = get_AS(loaded_model, feature_slide)
                signal = get_signal()

                result = {
                    'data': slide,
                    'feature': feature_window,
                    'AS': awake_score,
                    'status': status,
                    'signal': signal,
                    'second': second,
                    'probabilities': probabilities,
                }

                q.put(result)


    # Close the serial port
    print("DONE")
    file.close()


    return 

def start_thread(path, time_rec, port):
    q = queue.Queue()
    t = threading.Thread(target=collectData, args=(path, time_rec, port, q))
    t.start()
    return q, t


