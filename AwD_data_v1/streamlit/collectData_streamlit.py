import numpy as np
import serial
import random
import pandas as pd
import scipy as sp
import queue
import threading
from extract_feature import FeatureExtract, STFT_feature
import pickle
import time 
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Process, Queue

def get_AS(model, feature):
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

# def process_data(slide, model, second ,q):
#     # Tính toán đặc trưng
#     feature_window = FeatureExtract(slide)
    
#     # Dự đoán từ mô hình
#     awake_score, status, probabilities = get_AS(model, feature_window)

#     signal = get_signal()

#     # Tính toán STFT
#     stft_features = STFT_feature(slide)

#     result = {
#         'data': slide,
#         'feature': feature_window,
#         'AS': awake_score,
#         'status': status,
#         'signal': signal,
#         'second': second,
#         'probabilities': probabilities,
#         'STFT': stft_features,
#     }

#     q.put(result)




def collectData(path, time_rec, port, q: queue):
    if serial.Serial:
        serial.Serial().close()
    # Open the serial port
    s = serial.Serial(port, baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
    file = open(path, "w")

    x = 0  # iterator of sample
    k = 15  # window
    sample_rate = 512
    window_size = k * sample_rate
    y = deque([0]* window_size, maxlen=window_size)


    model_path = 'gradient.pkl'

    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    print("START!")
    start = time.time()
    while x < (time_rec * 512):
        if x % 512 == 0:
            second = x//512
            print(second)
            end = time.time()
            print("Time: ", end-start)
        x += 1
        data = s.readline().decode('utf-8').rstrip("\r\n")
        file.write(str(data))
        file.write('\n')

        try:
            # print(data)
            y.append(int(data))

        except:
            print("Error", data)
            continue

        if x >= window_size:
            if x % (1 * sample_rate) == 0:
                slide = np.array(y, dtype=int)       

                feature_window = FeatureExtract(slide)

                feature_slide = [np.array(list(feature_window.values()))]

                awake_score, status, probabilities = get_AS(loaded_model, feature_slide)
                signal = get_signal()

                stft_features = STFT_feature(slide)


                result = {
                    'data': slide,
                    'feature': feature_window,
                    'AS': awake_score,
                    'status': status,
                    'signal': signal,
                    'second': second,
                    'probabilities': probabilities,
                    'STFT': stft_features,
                }


                q.put(result)



    # Close the serial port
    print("DONE")
    s.close()
    file.close()


    return 



# def collectData(path, time_rec, port, q: queue):
#     if serial.Serial:
#         serial.Serial().close()
#     # Open the serial port
#     s = serial.Serial(port, baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
#     file = open(path, "w")

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         x = 0  # iterator of sample
#         k = 15  # window
#         sample_rate = 512
#         window_size = k * sample_rate
#         y = deque([0]* window_size, maxlen=window_size)


#         model_path = 'gradient.pkl'

#         with open(model_path, 'rb') as model_file:
#             loaded_model = pickle.load(model_file)

#         print("START!")
#         start = time.time()
#         while x < (time_rec * 512):
#             if x % 512 == 0:
#                 second = x//512
#                 print(second)
#                 end = time.time()
#                 print("Time: ", end-start)
#             x += 1
#             data = s.readline().decode('utf-8').rstrip("\r\n")
#             file.write(str(data))
#             file.write('\n')

#             try:
#                 # print(data)
#                 y.append(int(data))

#             except:
#                 print("Error", data)
#                 continue

#             if x >= window_size:
#                 if x % (1 * sample_rate) == 0:
#                     slide = np.array(y, dtype=int) 
#                     # write_buffer = slide[-512:]
#                     # file.writelines(f"{item}\n" for item in write_buffer)

#                     executor.submit(process_data, slide, loaded_model,second,q)
#                     active_processes = len(multiprocessing.active_children())
#                     print(f"Active processes: {active_processes}")

#                     # result = {
#                     #     'data': slide,
#                     #     'feature': feature_window,
#                     #     'AS': awake_score,
#                     #     'status': status,
#                     #     'signal': signal,
#                     #     'second': second,
#                     #     'probabilities': probabilities,
#                     #     'STFT': stft_features,
#                     # }



#     # Close the serial port
#     print("DONE")
#     s.close()
#     file.close()


    return 

def start_thread(path, time_rec, port):
    q = Queue()
    p = Process(target=collectData, args=(path, time_rec, port, q))
    p.start()
    return q, p


