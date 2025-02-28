import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib.patches import Wedge
from matplotlib.patches import Arc
import matplotlib.animation as animation
from PIL import Image, ImageSequence, ImageTk


def create_image(data, diction, image_path, awake_score):
    # matplotlib.use("TkAgg")
    plt.ion()
    # Tạo hình ảnh chính và các hình ảnh con
    fig = plt.figure(figsize=(10, 5))
    # Plot raw
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data)
    ax1.set_title('EEG Raw Values')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('RawValue')
    ax1.set_ylim(-256, 256)

    # # Plot STFT
    gif_path = "AD_Code/vibrate.gif"  # Thay bằng đường dẫn đến file GIF
    img = Image.open(gif_path)

    # Lấy tất cả các frame từ GIF
    frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis("off")
    im = ax2.imshow(frames[0])

    # Hàm cập nhật frame
    def update(frame):
        im.set_array(frame)
        return [im]

    # Tạo animation
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    # Plot brainwave
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(diction['delta'], label="delta")
    ax3.plot(diction['theta'], label="theta")
    ax3.plot(diction['alpha'], label="alpha")
    ax3.plot(diction['beta'], label="beta")
    ax3.set_title('Frequency Bands')
    ax3.set_xlabel('Time [sec]')
    ax3.set_ylabel('Power')
    ax3.set_ylim(0, 400)
    ax3.legend()

    # Circular Progress Bar (viền ngoài)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title("Progress Bar")
    ax4.set_aspect('equal')  # Giữ tỉ lệ hình tròn không bị méo

    progress = awake_score  # Tiến trình % (0-100)
    start_angle = 90  # Góc bắt đầu (từ trên cùng)
    end_angle = 90 - 360 * (progress / 100)  # Quay theo chiều kim đồng hồ

    # Vẽ vòng tròn nền 
    bg_circle = Arc((0, 0), 1.6, 1.6, angle=0, theta1=0, theta2=360, color="green", linewidth=8)
    ax4.add_patch(bg_circle)

    # Vẽ vòng tròn tiến trình (màu xanh)
    progress_arc = Arc((0, 0), 1.6, 1.6, angle=0, theta1=start_angle, theta2=end_angle, color="white", linewidth=8)
    ax4.add_patch(progress_arc)

    # Hiển thị phần trăm tiến trình
    ax4.text(0, 0, f"{progress}%", ha='center', va='center', fontsize=14, fontweight='bold')

    # Hiển thị hình ảnh
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def Frequency(data):
    FFT_signal = sp.signal.stft(data, 512, nperseg=15 * 512, noverlap=14 * 512)
    f, t, Zxx = FFT_signal
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


if __name__ == "__main__":
    data = np.array([1,2,3,4,5,6,7,8]*64*15)
    print(data.shape)
    diction, FFT_signal = Frequency(data)
    image_path = 'test.png'
    create_image(data, diction, image_path)