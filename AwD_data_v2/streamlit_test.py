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
from collectData_streamlit_demo import start_thread
# from collectData_streamlit import start_thread


plt.ion()

st.set_page_config(
    page_title = "Data Acquisition",
    layout = 'wide'
)

plt.tight_layout() # thêm vào từng hàm tạo figure matplotlib

st.title("Data Acquisition | Real-time Dashboard")


def create_progress_bar(percentage: int):
    fig = go.Figure()

    green_width = percentage
    red_width = 100 - percentage

    fig.add_trace(go.Bar(
        x=[green_width],
        y=[0],
        orientation='h',
        marker=dict(color='green'),
        width=[0.1],
        name='Progress',
        text=[f'{percentage}%'],  
        textposition='inside',   
        insidetextanchor='middle' 
    ))

    fig.add_trace(go.Bar(
        x=[red_width],
        y=[0],
        orientation='h',
        marker=dict(color='red'),
        width=[0.1],
        name='Remaining'
    ))

    fig.update_layout(
        barmode='stack',
        # title=f'Progress: {percentage}%',
        xaxis_title="Percentage",
        yaxis=dict(showticklabels=False),
        xaxis=dict(range=[0, 100]),
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig



def create_bar_chart(probabilities: list[int]):
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']
    
    fig = go.Figure([go.Bar(
        x=tasks,
        y=probabilities,
        text=[f'{p*100:.0f}%' for p in probabilities],  
        textposition='outside', 
        marker=dict(color='lightgreen')
    )])

    fig.update_layout(
        title='Probability of Tasks',
        xaxis_title='Tasks',
        yaxis_title='Probability (%)',
        template="plotly_dark"
    )
    
    return fig

# Hàm vẽ Raw EEG
def create_raw_eeg(data: list[int]):
    fig = go.Figure(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='EEG Raw Values',
        xaxis_title='Samples',
        yaxis_title='RawValue',
        template="plotly_dark",
        yaxis=dict(range=[-256, 256])
    )

    return fig


# Hàm vẽ STFT
def create_STFT(stft_features: tuple):
    t, f, Zxx = stft_features
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(Zxx),
        x=t,
        y=f,
        colorscale='Viridis',
        colorbar=dict(title='Magnitude'),
        zmin=-1,
        zmax=5
    ))

    fig.update_layout(
        title='STFT Magnitude',
        xaxis_title='Time [sec]',
        yaxis_title='Frequency [Hz]',
        template="plotly_dark",
        yaxis=dict(range=[0.5, 40])
    )

    return fig

def create_band_power_chart(diction: dict):
    fig = go.Figure()

    # Chuyển đổi deque thành list hoặc numpy array
    fig.add_trace(go.Scatter(
        x=list(range(len(diction['delta']))),
        y=list(diction['delta']),  # Chuyển deque thành list
        mode='lines',
        name='Delta',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(diction['theta']))),
        y=list(diction['theta']),  # Chuyển deque thành list
        mode='lines',
        name='Theta',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(diction['alpha']))),
        y=list(diction['alpha']),  # Chuyển deque thành list
        mode='lines',
        name='Alpha',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(diction['beta']))),
        y=list(diction['beta']),  # Chuyển deque thành list
        mode='lines',
        name='Beta',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Frequency Bands',
        xaxis_title='Time [sec]',
        yaxis_title='Power',
        template="plotly_dark",
        yaxis=dict(range=[0, 400]),
        showlegend=True
    )

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
        name="Trend Chart",
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
        showlegend=False,
        width=800,  # Đặt chiều rộng của biểu đồ
        height=400  # Đặt chiều cao của biểu đồ
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


def create_dashboard(status: str, signals: deque, probabilities: list[int], data: list[int], diction: dict, awake_score_1m: deque, tick: list[int], second: int, stft_features: tuple):
    first_col, second_col = st.columns([2,1])

    with first_col:
        as_col, status_col, signal_col, _ = st.columns([1,1,1,1])

        as_col.metric(label="Awake Score", value=round(awake_score_1m[-1]), delta = round(awake_score_1m[-1]) - round(awake_score_1m[-2]))
        status_col.metric(label="Status", value=status)
        signal_col.metric(label="Signal", value=round(signals[-1]), delta = round(signals[-1]) - round(signals[0]))


        as_bar, prob_col = st.columns([1, 1])

        with as_bar:
            fig = create_progress_bar(awake_score_1m[-1])
            st.plotly_chart(fig, key=f"as_{second}")

        with prob_col:
            fig = create_bar_chart(probabilities)
            st.plotly_chart(fig, key=f"probabilities_{second}")

        raw_col, band_col = st.columns([0.5, 0.5])
        with raw_col:
            fig = create_raw_eeg(data)
            st.plotly_chart(fig, key=f"raw_{second}")

        with band_col:
            fig = create_band_power_chart(diction)
            st.plotly_chart(fig, key=f"band_{second}")
    
    # with spec_col:
    #     pass

    with second_col:
        minute = int(second // 60)
        sec = int(second % 60)
        second_col.metric(label="Time Recorded", value=f"{minute:02}:{sec:02}")
        # st.write(f"Time Recorded: {minute:02}:{sec:02}")

        stft_fig = create_STFT(stft_features)
        st.plotly_chart(stft_fig, key=f"stft_{second}")

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
            # print("Queue is not empty: ", data_queue.qsize())
            data_dict = data_queue.get()

            raw_eeg_data, features, awake_score, status, signal, second, probabilities, stft_features = tuple(data_dict.values())
            
            for i, band in enumerate(band_power):
                band_power[band].append(list(features.values())[i])


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
                    second = second,
                    stft_features = stft_features
                )

def get_info():

    first_name = st.text_input("First name:")
    last_name = st.text_input("Last name:")

    # Nhập tuổi
    age = st.number_input("Your age:", min_value=0, max_value=120, step=1)

    return first_name, last_name, age

info_placeholder = st.empty()
with info_placeholder.container():
    first_name, last_name, age = get_info()


if st.button('Start Recording'):
    info_placeholder.empty()
    port = "/dev/tty.usbserial-1220"
    path = f"{first_name}_{age}.txt"
    time_rec = int(40*60)
    with st.spinner('Recording...'):
        main(port, path, time_rec)

    if st.button("Stop Recording"):
        quit()







     






