from matplotlib import pyplot as plt
import cmath
import numpy as np
from scipy.signal import butter, lfilter, freqz
import pandas as pd
import threading
import plotFig as pf

thread_lock = threading.Lock()

timer=0
slide_window_speed=1    #单位：秒
window_length=20 #窗口长度：20s  呼吸频率：0.34

xData,yData=[],[]

def slide_window(phase):
    while True:
        global timer    #时间序号,即每次移动几秒的数据
        total_length=len(phase)
        if window_length+timer>=total_length:
            #python不需要手动回收global
            break   #用whilt true和break来终止函数，从而终止线程
        else:
            phase_window=phase[timer:timer+window_length]
            timer+=slide_window_speed*100
            return phase_window

def data_update_thread(phase):
    global xData,yData
    phase=slide_window(phase)
    #将phase_window按plotFig中的方案预处理
    if pf.ifLimitTimeRange:
        phase,axisTime=pf.limit_time_range(phase,axisTime,pf.time_begin,pf.time_end)
        print("len(data):",len(phase))

    if pf.ifMinusMean:
        phase=pf.minus_mean(phase)

    if pf.ifFilter:
        phase=pf.butter_lowpass_filter(phase,pf.cutoff,fs=100)

    xData,yData=pf.fft(phase,axisTime)

# ad_rdy_ev = threading.Event()
# ad_rdy_ev.set()  # 设置线程运行
# t = threading.Thread(target=data_update, args=()) # 更新数据，参数说明：target是线程需要执行的函数，args是传递给函数的参数）
# t.daemon = True
# t.start()  # 线程执行

# plt.show() # 显示图像



if __name__ == "__main__":
    phase,axisTime,amp=pf.read_data(pf.filename)
    window_length=pf.window_length

    # thread1 = threading.Thread(target=data_update_thread(phase,window_length,timer), name='Thread 1')
    # thread2 = threading.Thread(target=pf.plot_fig(xData,yData), name='Thread 2')
    
    thread1 = threading.Timer(slide_window_speed, data_update_thread(phase))
    thread2 = threading.Timer(slide_window_speed, pf.plot_fig(xData,yData))

    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()




