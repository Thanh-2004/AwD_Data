import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import plotly.graph_objects as go
from collections import deque
import queue
import threading
# from collectData_streamlit_demo import start_thread
from collectData_streamlit import start_thread



plt.ion()

st.set_page_config(
    page_title = "Data Acquisition",
    # layout = 'wide'
)

plt.tight_layout() # thêm vào từng hàm tạo figure matplotlib

st.title("Data Acquisition | Real-time Dashboard")

def create_progress_bar(percentage: int):
    fig, ax = plt.subplots(figsize=(10, 1))
    
    green_width = percentage
    red_width = 100 - percentage
    
    ax.barh(0, green_width, color='green', height=0.1)
    ax.barh(0, red_width, left=green_width, color='red', height=0.1)
    ax.vlines(green_width, ymin=-0.1, ymax=0.1, color='black', linewidth=2)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 0.2)
    ax.axis('off')  
    
    ax.text(green_width - 3, 0, f'{percentage}%', color='white', ha='center', va='center')
    
    return fig

def create_bar_chart(probabilities: list[int]):
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tasks, probabilities, color='lightgreen') 
    ax.set_title('Probability of Tasks')
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Probability (%)')
    
    return fig

def create_raw_eeg(data: list[int]):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data)
    ax1.set_title('EEG Raw Values')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('RawValue')
    ax1.set_ylim(-256, 256)

    return fig

def create_band_power_chart(diction: dict):
    fig, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(diction['delta'], label="delta")
    ax3.plot(diction['theta'], label="theta")
    ax3.plot(diction['alpha'], label="alpha")
    ax3.plot(diction['beta'], label="beta")
    ax3.set_title('Frequency Bands')
    ax3.set_xlabel('Time [sec]')
    ax3.set_ylabel('Power')
    ax3.set_ylim(0, 400)
    ax3.legend()

    return fig

def generate_minute_candlestick_data(prev_data, awake_score_1m: deque):
    if prev_data is not None:
        last_timestamp = prev_data['Date'].iloc[-1] + pd.Timedelta(seconds=60)  
    else:
        last_timestamp = pd.to_datetime("2023-01-01 00:00:00")

    open = awake_score_1m[0]
    close = awake_score_1m[-1]
    high = max(awake_score_1m)
    low = min(awake_score_1m)

    new_data = pd.DataFrame({
        'Date': [last_timestamp],
        'Open': [open],
        'High': [high],
        'Low': [low],
        'Close': [close]
    })

    if prev_data is not None:
        prev_data = prev_data.append(new_data, ignore_index=True)
    else:
        prev_data = new_data

    return prev_data

def plot_candlestick(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%H:%M')
    df['Date'] = pd.to_datetime(df['Date'])

    df['MA60'] = df['Close'].rolling(window=60).mean()

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name="Trend Chart"
    )])

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA60'],
        mode='lines',
        name='MA 60',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Awake Score Trends (1 Minute)',
        xaxis_title='Time',
        yaxis_title='Awake Score',
        xaxis_rangeslider_visible=False,  
        template="plotly_dark",  
        dragmode="zoom",  
        showlegend=False
    )
    
    return fig

def create_candle(awake_score_1m, second):
    print("Current second:", second)
    if 'data' not in st.session_state:
        st.session_state.data = generate_minute_candlestick_data(None, awake_score_1m)

    elif second % 60 == 0:
        st.session_state.data = generate_minute_candlestick_data(st.session_state.data, awake_score_1m)

    fig = plot_candlestick(st.session_state.data)

    return fig


def create_dashboard(status: str, signals: deque, probabilities: list[int], data: list[int], diction: dict, awake_score_1m: deque, tick: list[int], second: int):
    status_col, signal_col, _, time_recorded = st.columns([1,1,1,1])

    status_col.metric(label="Status", value=status)
    signal_col.metric(label="Signal", value=round(signals[-1]), delta = round(signals[-1]) - round(signals[0]))

    with time_recorded:
        minute = int(second // 60)
        sec = int(second % 60)
        st.write(f"Time Recorded: {minute:02}:{sec:02}")

    as_bar, prob_col = st.columns([1, 1])

    with as_bar:
        as_bar.metric(label="Awake Score", value=round(awake_score_1m[-1]), delta = round(awake_score_1m[-1]) - round(awake_score_1m[-2]))
        fig = create_progress_bar(awake_score_1m[-1])
        st.pyplot(fig)

    with prob_col:
        fig = create_bar_chart(probabilities)
        st.pyplot(fig)

    raw_col, band_col = st.columns([0.5, 0.5])
    with raw_col:
        fig = create_raw_eeg(data)
        st.pyplot(fig)

    with band_col:
        fig = create_band_power_chart(diction)
        st.pyplot(fig)
    
    # with spec_col:
    #     pass

    fig = create_candle(tick, second)
    st.plotly_chart(fig, use_container_width=True, key=f"candlestick_{second}")

def main(port, path, time_rec):
    print("Test main")
    # port = "COM3"
    # port = "/dev/tty.usbserial-1220"
    # path = "test.txt"
    # time_rec = int(40*60)


    awake_score_1m = deque([0] * 60, maxlen=60)
    tick = [0]*60
    signals = deque([0]*2, maxlen=2)
    placeholder = st.empty()
    band_power = {
        "alpha": deque([0]*60, maxlen=60),
        "beta": deque([0]*60, maxlen=60),
        "theta": deque([0]*60, maxlen=60),
        "delta": deque([0]*60, maxlen=60)
    }

    data_queue, data_thread = start_thread(path, time_rec, port)
    # data_thread.join()
    print("Thread Checking")

    while True:
        # print("Loop checking")
        if not data_queue.empty():
            print("Queue is not empty: ", data_queue.qsize())
            data_dict = data_queue.get()

            raw_eeg_data, features, awake_score, status, signal, second, probabilities = tuple(data_dict.values())
            
            for i, band in enumerate(band_power):
                band_power[band].append(list(features.values())[i])


            # awake_score = np.random.randint(0,100)
            # status = np.random.choice(["Awake", "Fatigue"])
            # signal = np.random.randint(0, 100)
            # probabilities = list(np.random.randint(0, 101, size=5))
            # raw_eeg_data = list(np.random.randint(-256, 256, size=256))
            # band_power = {
            #     'delta': list(np.random.randint(0, 400, size=60)),
            #     'theta': list(np.random.randint(0, 400, size=60)),
            #     'alpha': list(np.random.randint(0, 400, size=60)),
            #     'beta': list(np.random.randint(0, 400, size=60))
            # }

            # if second == 15:
            #     awake_score_1m.append(0)
            #     signals.append(0)

            awake_score_1m.append(awake_score)
            signals.append(signal)

            if len(awake_score_1m) == 60:
                tick = awake_score_1m

            # Test hàm dashboard
            with placeholder.container():
                create_dashboard(
                    status=status,
                    signals=signals,
                    probabilities=probabilities,
                    data=raw_eeg_data,
                    diction=band_power,
                    awake_score_1m=awake_score_1m,
                    tick = tick, 
                    second = second
                )

def get_info():

    first_name = st.text_input("First name:")
    last_name = st.text_input("Last name:")

    # Nhập tuổi
    age = st.number_input("Nhập tuổi của bạn:", min_value=0, max_value=120, step=1)

    return first_name, last_name, age

info_placeholder = st.empty()
with info_placeholder.container():
    first_name, last_name, age = get_info()


if st.button('Start Recording'):
    info_placeholder.empty()
    port = "/dev/tty.usbserial-11220"
    path = f"{first_name}_{age}.txt"
    time_rec = int(40*60)
    with st.spinner('Recording...'):
        main(port, path, time_rec)

    if st.button("Stop Recording"):
        quit()




# if __name__ == "__main__":
#     awake_score_1m = deque(maxlen=60)
#     tick = [0]*60
#     signals = deque(maxlen=2)
#     placeholder = st.empty()

#     for second in range (300):
#         print(second)
#         awake_score = np.random.randint(0, 101)
#         status = np.random.choice(["Awake", "Fatigue"])

#         signal = np.random.randint(0, 100)

#         probabilities = list(np.random.randint(0, 101, size=5))

#         raw_eeg_data = list(np.random.randint(-256, 256, size=256))

#         band_power = {
#             'delta': list(np.random.randint(0, 400, size=60)),
#             'theta': list(np.random.randint(0, 400, size=60)),
#             'alpha': list(np.random.randint(0, 400, size=60)),
#             'beta': list(np.random.randint(0, 400, size=60))
#         }

#         if second == 0:
#             awake_score_1m.append(0)
#             signals.append(0)

#         awake_score_1m.append(awake_score)
#         signals.append(signal)

#         if len(awake_score_1m) == 60:
#             tick = awake_score_1m

#         # Test hàm dashboard
#         with placeholder.container():
#             create_dashboard(
#                 status=status,
#                 signals=signals,
#                 probabilities=probabilities,
#                 data=raw_eeg_data,
#                 diction=band_power,
#                 awake_score_1m=awake_score_1m,
#                 tick = tick, 
#                 second = second
#             )
#             time.sleep(1)





     






