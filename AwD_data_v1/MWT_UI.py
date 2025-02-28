import tkinter as tk
import time
import random
import collectData 
import collectData_demo

import os
import threading
from plot import *


class ExperimentApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Thí nghiệm Sóng não")
        self.root.geometry("1200x800")
        self.root.bind('<space>', self.on_spacebar_press)
        self.root.bind('<Return>', self.on_enter_press)
        
        self.current_frame = None

        self.count = 0
        
        self.create_login_screen()
    
    def on_spacebar_press(self, event):
        if self.current_frame:
            # Gọi hàm tiếp theo dựa trên giao diện hiện tại
            if hasattr(self, 'next_function'):
                self.next_function()

    def on_enter_press(self, event):
        if self.current_frame:
            self.logged()
        self.root.unbind('<Return>')

    def create_login_screen(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Đăng nhập").pack()
        
        self.name_var = tk.StringVar()
        tk.Label(frame, text="Tên:").pack()
        tk.Entry(frame, textvariable=self.name_var).pack()
        
        self.age_var = tk.StringVar()
        tk.Label(frame, text="Tuổi:").pack()
        tk.Entry(frame, textvariable=self.age_var).pack()
        
        self.gender_var = tk.StringVar()
        tk.Label(frame, text="Giới tính:").pack()
        tk.Entry(frame, textvariable=self.gender_var).pack()
        
        self.address_var = tk.StringVar()
        tk.Label(frame, text="Địa chỉ:").pack()
        tk.Entry(frame, textvariable=self.address_var).pack()
        
        self.phone_var = tk.StringVar()
        tk.Label(frame, text="Số điện thoại:").pack()
        tk.Entry(frame, textvariable=self.phone_var).pack()
        
        tk.Button(frame, text="Đăng nhập", command=self.create_intro_screen).pack(pady=10)
        
        if not os.path.exists(f"Data/{self.name_var}"):
            os.makedirs(f"Data/{self.name_var}")

        self.logged = self.create_intro_screen
        self.current_frame = frame

    def create_intro_screen(self):
        self.clear_frame()
        
        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Giới thiệu về dự án").pack()
        tk.Label(frame, text="Chào mừng bạn đến với dự án XYZ. Hãy bấm phím [spacebar] để tiếp tục.").pack(pady=20)

        self.next_function = self.create_general_instructions
        self.current_frame = frame
    
    def create_general_instructions(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Hướng dẫn chung").pack()
        tk.Label(frame, text="Trước khi bắt đầu, hãy đọc kỹ hướng dẫn sau đây. Bấm phím [spacebar] để tiếp tục.").pack(pady=20)

        self.next_function = self.create_task1_instructions
        self.current_frame = frame
    
    def create_task1_instructions(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 1 - Hướng dẫn").pack()
        tk.Label(frame, text="Giới thiệu về tác vụ 1. Hướng dẫn người dùng thực hiện. Bấm phím [spacebar] khi đã sẵn sàng.").pack(pady=20)

        self.next_function = self.create_task1_execution
        self.current_frame = frame

    def create_task1_execution(self):
        self.clear_frame()


        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Tác vụ 1 - Thực hiện").pack()
        tk.Label(frame, text="Hiển thị màn hình trắng\n(Hiển thị hình ảnh sóng não thô theo thời gian thực)").pack(pady=20)

        # Load hình ảnh ban đầu
        self.img = Image.open(white_image)  # Đổi thành file ảnh phù hợp
        self.img = self.img.resize((800, 600))
        self.photo = ImageTk.PhotoImage(self.img)

        # Thay canvas bằng Label để hiển thị ảnh
        self.img_label = tk.Label(frame, image=self.photo, bg="white")
        self.img_label.pack()
        
        self.task1_timer = tk.StringVar()
        self.task1_timer.set("Thời gian: 0:30")

        tk.Label(frame, textvariable=self.task1_timer).pack()

        self.current_frame = frame

        current_time = self.task1_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        time_rec = int(minutes * 60 + seconds)
        path = f"Data/{self.name_var}/task1.txt"
        data_thread = threading.Thread(target=collectData_demo.collectData, args=(path, time_rec, port, image_path, root))
        data_thread.start()

        self.root.after(1000, self.update_task1_timer)

    def update_task1_timer(self):
        ## Update Dashboard
        if self.count >= 15:
            self.img = Image.open(image_path)  
            self.img = self.img.resize((800, 600))
            self.photo = ImageTk.PhotoImage(self.img)
            self.img_label.configure(image=self.photo)
            self.img_label.image = self.photo  
        else:
            self.img = Image.open(white_image)  
            self.img = self.img.resize((800, 600))
            self.photo = ImageTk.PhotoImage(self.img)
            self.img_label.configure(image=self.photo)
            self.img_label.image = self.photo  

        ## Update Timer
        current_time = self.task1_timer.get().split(": ")
        minutes, seconds = map(int, current_time[1].split(":"))
        seconds -= 1
        self.count += 1
        if seconds == -1:
            minutes -= 1
            seconds = 59
        self.task1_timer.set(f"Thời gian: {minutes}:{seconds:02d}")
        if minutes == 0 and seconds == 0:
            tk.Label(self.current_frame, text="Đã kết thúc Tác vụ 1. \n Bấm phím [spacebar] để chuyển sang tác vụ tiếp theo.").pack(pady=30)
            self.next_function = self.create_end_screen 
        else:
            self.root.after(1000, self.update_task1_timer)

    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

    def create_end_screen(self):
        self.clear_frame()

        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        tk.Label(frame, text="Kết thúc quá trình đo thí nghiệm").pack()
        tk.Label(frame, text="Cảm ơn bạn đã tham gia. Các file dữ liệu đã đo được liệt kê dưới đây.\nCác file dữ liệu đã nằm trong folder Data -> {}".format(self.name_var.get())).pack(pady=20)
        
        # files = ["data_{}_{}.txt".format(self.name_var.get(), self.age_var.get())]  # Example filename
        # for file in files:
        #     tk.Label(frame, text=file).pack()

        self.current_frame = frame


if __name__ == "__main__":
    port = "COM3"
    root = tk.Tk()
    app = ExperimentApp(root)
    image_path = "test.png"
    white_image = "white.jpeg"
    count = 0
    root.mainloop()
