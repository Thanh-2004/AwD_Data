import numpy as np
from plot import *
import random
import time

def FeatureExtract(data):
    f, t, Zxx = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
    delta = np.array([], dtype=float)
    theta = np.array([], dtype=float)
    alpha = np.array([], dtype=float)
    beta = np.array([], dtype=float)
    for i in range(0, int(t[-1])):
        indices = np.where((f >= 0.5) & (f <= 4))[0]
        delta = np.append(delta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 4) & (f <= 8))[0]
        theta = np.append(theta, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 8) & (f <= 13))[0]
        alpha = np.append(alpha, np.sum(np.abs(Zxx[indices, i])))

        indices = np.where((f >= 13) & (f <= 30))[0]
        beta = np.append(beta, np.sum(np.abs(Zxx[indices, i])))

    abr = alpha / beta
    tbr = theta / beta
    dbr = delta / beta
    tar = theta / alpha
    dar = delta / alpha
    dtabr = (alpha + beta) / (delta + theta)

    diction = {"delta": delta,
               "theta": theta,
               "alpha": alpha,
               "beta": beta,
               "abr": abr,
               "tbr": tbr,
               "dbr": dbr,
               "tar": tar,
               "dar": dar,
               "dtabr": dtabr
               }
    return diction

def Freq_feature(slide):
    test_slide = pd.DataFrame.from_dict(FeatureExtract(slide)).values
    # test_slide = scaler.fit_transform(test_slide)
    # print(test_slide.shape)
    return test_slide

def get_AS(model=None, feature=None):
    awake_score = random.randint(0,100)
    return awake_score


def collectData(path, time_rec, port, image_path, plot=True):
    path = "/Users/nguyentrithanh/Documents/Lab/EEG_AwakeDrive/AD_Code/test_data/BachCalm.txt"
    file = open(path, "r")

    x = 0  # iterator of sample
    y = np.array([], dtype=int)  # value
    k = 15  # window
    sample_rate = 512
    window_size = k * sample_rate
    feature = []

    print("START!")
    while x < (time_rec * 512):
        if x % 512 == 0:
            print(x // 512)
            time.sleep(0.95)

        x += 1
        data = file.readline()
        # data = s.readline().decode('utf-8').rstrip("\r\n")
        # file.write(str(data))
        # file.write('\n')
        y = np.append(y, int(data))

        if x >= window_size:
            if x % (1 * sample_rate) == 0:
                sliding_window_start = x - window_size
                sliding_window_end = x
                slide = np.array(y[sliding_window_start:sliding_window_end]) 
                feature_window = FeatureExtract(slide)
                feature.append(feature_window)

                freq_feature = Freq_feature(slide)
                print("Freq: ", freq_feature.shape)
                awake_score = get_AS()

                print(awake_score)
                create_image(slide, feature_window, image_path, awake_score)



    # Close the serial port
    print("DONE")
    file.close()


    return 


