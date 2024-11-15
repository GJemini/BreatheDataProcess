from matplotlib import pyplot as plt
import cmath
import numpy as np
from scipy.signal import butter, lfilter, freqz
import pandas as pd
import threading

#pwd:D:\StudyFiles\ILoveStudy\pro\BreatheDetect-master\data

class BreatheDataProcess:
    def __init__(self):
        #判断是否做相位补偿、是否滤波、是否消除0频信号,选择cir信道
        self.ifFilter=1
        self.cutoff=1

        self.ifLimitTimeRange=0
        self.time_begin,self.time_end=10,50

        self.ifPhaseCompensate=0
        self.ifMinusMean=1
        self.elseAdd=0

        self.ifPrintMax=1
        self.ifDebug=0

        self.ifSlideWindow=0
        self.window_length=20 #窗口长度：20s  呼吸频率：0.34
        self.window_moving_speed=1 #窗口每一秒更新一次

        self.cir=0
        self.filename = 'data_moving.txt'
        self.stepTime=0.01 #假设每个数据对应0.01s

        #缓冲区
        self.xData1,self.yData1=[],[]
        self.xData2,self.yData2=[],[]
        self.phase,self.axisTime,self.amp=[],[],[]
        self.timer=0
        self.lock = threading.Lock()
        self._debug=0
        self.lock=threading.Lock()
        self.is_recorded=False  #观察线程是否有序交替运行

        self.read_data(self.filename)#在初始化中读取数据

    #巴特沃斯低通滤波：设置参数
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a
    ##巴特沃斯低通滤波：调用函数
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        y = lfilter(b, a, data)
        return y    

    #选取某段时间的数据；time_end=-1表示不限制end
    def limit_time_range(self,data,axis,time_begin,time_end):
        #TODO:缺少perror
        if time_end==-1:
            data=data[time_begin*100:] #if (0<=time_begin<time_end and time_end<data.size()) else -1
            axis=self.axisTime[time_begin*100:]
        else:
            data=data[time_begin*100:time_end*100]
            axis=self.axisTime[time_begin*100:time_end*100]
        return data,axis

    def read_data(self,filename):
        # 相比open(),with open()不用手动调用close()方法
        with open(filename, 'r') as f:
            next(f)#跳过第一行的数据
            lines = f.readlines() 
            #数据格式：第一行乱码 第二行起为：F7188 R0 1463 -749 -558 2044...长度为82；CIR共有0-39共40个信道
            tempTime=0
            linePhase,lastOriginalPhase=100,100
            flag=0 #flag=n表示当前数据在(n*pi-pi,n*pi+pi)的区间内
            for line in lines:
                lineData=line.split()#按空格分割，lineData.length=40*2+2=82
                if len(lineData)>=82:#防止最后一行数据不全
                    x,y=int(lineData[2*self.cir+2]),int(lineData[2*self.cir+3])#取第n个cir信道的数据
                    z = complex(x, y)
                    lineAmp = abs(z)
                    mutationLimit=6
                    #相位补偿：用原始数据的前后两帧判断是否发生突变，用flag标示当前区间
                    if self.ifPhaseCompensate:
                        if linePhase==100:#初始值
                            linePhase = cmath.phase(z)
                            lastOriginalPhase=linePhase
                        else:
                            tempPhase=cmath.phase(z)
                            differ=tempPhase-lastOriginalPhase
                            if abs(differ)<mutationLimit:#未突变，继续当前flag区间的加减
                                linePhase=tempPhase+flag*2*cmath.pi
                                lastOriginalPhase=tempPhase
                            elif tempPhase>0:#突变，且当前帧超过下界，flag--
                                flag-=1
                                linePhase=tempPhase+flag*2*cmath.pi
                                lastOriginalPhase=tempPhase
                            elif tempPhase<0:#突变，且当前帧超过上界，flag++
                                flag+=1
                                linePhase=tempPhase+flag*2*cmath.pi
                                lastOriginalPhase=tempPhase
                    else:
                        linePhase=cmath.phase(z)

                    self.amp.append(lineAmp)
                    self.phase.append(linePhase)
                    self.axisTime.append(tempTime)
                    tempTime+=self.stepTime
        # return phase,axisTime,amp 不需要返回值
        self.xData1=self.axisTime
        self.phase=np.unwrap(self.phase)
        self.yData1=self.phase

    def minus_mean(self,phase):
        mean=np.mean(phase)
        phase=phase-mean+self.elseAdd
        return phase

    #保存处理好的self.phase数据
    def save_processed_data(self):
        f=open("processed_data.txt","w")
        for line in self.phase:
            f.write(str(line)+'\n')
        f.close()

    def fft(self,data,axisTime):
        n=len(axisTime)
        frequencies = np.fft.fftfreq(n, self.stepTime)
        xData=frequencies[:n // 2]
        fft=np.fft.fft(data,n,axis=0)/n*2 #amp是幅值关于时间分布的数组
        phaseFFT=np.abs(fft)
        yData=phaseFFT[:n // 2]
        return xData,yData

    #画图，绘制x1-y1,x2-y2
    def plot_fig(self,xData1,yData1,xData2,yData2):
        with self.lock:
            fig = plt.figure(figsize=(10, 10))  # 创建绘图窗口，并设置窗口大小
            # 画第一张图,相位-时间
            ax1 = fig.add_subplot(211)  # 将画面分割为2行1列选第一个
            ax1.plot(xData1, yData1, 'blue', label='phase-time')
            #ax1.plot(self.axisTime, self.phase, 'blue', label='phase-time')
            #ax1.set_xlim(35,40)
            ax1.set_xlabel('time(s)')  # 设置X轴名称
            ax1.set_ylabel('phase')  # 设置Y轴名称
            # 画第二张图，相频特性
            ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
            #print(len(phaseFFT))
            #ax2.plot(frequencies, np.real(phaseFFT), 'blue', label='phase-frequence')
            ax2.plot(xData2, yData2, 'blue', label='phase-frequence')
            ax2.set_xlim(0,5)
            ax2.set_xlabel('frequence(Hz)')  # 设置X轴名称
            ax2.set_ylabel('phase')  # 设置Y轴名称
            if self.ifPrintMax:
                max_idx = np.argmax(yData2)
                max_x, max_y = xData2[max_idx], yData2[max_idx]
                plt.scatter(max_x, max_y, color='red', s=50)
                plt.annotate(f'max: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y),
                        xytext=(max_x+0.05, max_y-0.05))

            plt.show()  # 显示绘制的图
    #重构
    def plot_fig(self):
        with self.lock:
            fig = plt.figure(figsize=(10, 10))  # 创建绘图窗口，并设置窗口大小
            # 画第一张图,相位-时间
            ax1 = fig.add_subplot(211)  # 将画面分割为2行1列选第一个
            ax1.plot(self.xData1, self.yData1, 'blue', label='phase-time')
            ax1.set_xlabel('time(s)')  # 设置X轴名称
            ax1.set_ylabel('phase')  # 设置Y轴名称
            # 画第二张图，相频特性
            ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
            ax2.plot(self.xData2, self.yData2, 'blue', label='phase-frequence')
            ax2.set_xlim(0,5)
            ax2.set_xlabel('frequence(Hz)')  # 设置X轴名称
            ax2.set_ylabel('phase')  # 设置Y轴名称
            max_idx = np.argmax(self.yData2)
            max_x, max_y = self.xData2[max_idx], self.yData2[max_idx]
            if self.ifPrintMax:
                plt.scatter(max_x, max_y, color='red', s=50)
                plt.annotate(f'max: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y),
                        xytext=(max_x+0.05, max_y-0.05))

            plt.show()  # 显示绘制的图

    #数据处理，只保存xData和yData用于画图
    def total_process(self,axis,data):
        if self.ifLimitTimeRange:
            data,axis=self.limit_time_range(data,axis,self.time_begin,self.time_end)
            print("len(data):",len(data))

        if self.ifMinusMean:
            data=self.minus_mean(data)

        if self.ifFilter:
            data=self.butter_lowpass_filter(data,self.cutoff,fs=100)

        self.xData1,self.yData1=axis,data
        self.xData2,self.yData2=self.fft(data,axis)

    #找到并返回滑动窗口位置，timer++
    def find_silde_window(self,axis,data):
        while True:
            total_length=len(data)
            if self.window_length+self.timer>=total_length:
                return -1
                #break   #用whilt true和break来终止函数，从而终止线程
            else:
                phase_window=data[self.timer:self.timer+self.window_length]
                axis_window=axis[self.timer:self.timer+self.window_length]
                self.timer+=self.window_moving_speed*100
                self.xData1=axis_window
                self.yData1=phase_window
                return axis_window,phase_window
            
    #线程地更新xData,yData
    def data_update_thread(self,data,axis):
        while True:
            if self.find_silde_window(data,axis)==-1:
                print("Data ends.")
                break
            else:
                data_window,axis_window=self.find_silde_window(data,axis)
                #TODO:bug为axis_window并不从0开始
                #将phase_window按plotFig中的方案预处理
                self.total_process(data_window,axis_window)
                print("thread time:",self._debug,"data_length",len(self.xData2))
                self._debug+=1

    def plot_fig_thread(self,x,y):
        plt.ion()

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax1.set_xlabel('time(s)')  # 设置X轴名称
        ax1.set_ylabel('phase')  # 设置Y轴名称

        ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
        ax2.set_xlim(0,2)
        ax2.set_xlabel('frequence(Hz)')  # 设置X轴名称
        ax2.set_ylabel('phase')  # 设置Y轴名称

        # x1_window=x[0:self.window_length*100]
        # y1_window=y[0:self.window_length*100]
        # x2_window,y2_window=self.total_process(y1_window,y1_window)
        # ax1.plot(x1_window, y1_window, 'blue', label='phase-time')
        # # ax2.plot(x2_window, y2_window, 'blue', label='phase-frequence')
        # ax2.plot(self.xData2, self.yData2, 'blue', label='phase-frequence')

        for i in range(0, len(x)-self.window_length,10):
            x1_window=x[i:i+self.window_length*100]
            y1_window=y[i:i+self.window_length*100]
            self.total_process(x1_window,y1_window)
            # max_idx = np.argmax(y2_window)
            # max_x, max_y = x2_window[max_idx], y2_window[max_idx]
            # print("max_x=",max_x,"max_y=",max_y)
            ax1.clear()
            ax2.clear()
            ax2.set_xlim(0,2)
            ax1.plot(self.xData1, self.yData1, 'blue', label='phase-time')
            ax2.plot(self.xData2, self.yData2, 'blue', label='phase-frequence')
            # ax2.plot(self.xData2, self.yData2, 'blue', label='phase-frequence')
            if self.ifPrintMax:
                max_idx = np.argmax(self.yData2)
                max_x, max_y = self.xData2[max_idx], self.yData2[max_idx]
                plt.scatter(max_x, max_y, color='red', s=50)
                plt.annotate(f'max: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y),
                        xytext=(max_x+0.05, max_y-0.05))
            plt.pause(0.001)
        plt.ioff()
        plt.show()

    #最终的输出函数
    def output(self):
        self.total_process(self.phase,self.axisTime)#更新xData和yData
        self.plot_fig(self.xData1,self.yData1,self.xData2,self.yData2)

    #最终的输出函数，用于滑动窗口
    def output_thread(self):
        # thread1 = threading.Timer(self.window_moving_speed, 
        #                           self.data_update_thread(self.phase,self.axisTime))
        thread2 = threading.Timer(self.window_moving_speed, 
                                  self.plot_fig_thread(self.axisTime,self.phase))


        # thread1.start()
        thread2.start()
        
        # thread1.join()
        thread2.join()
        
if __name__=="__main__":
    test=BreatheDataProcess()
    test.output_thread()

    # test.total_process(test.phase[2000:4000],test.axisTime[2000:4000])
    # test.plot_fig()



  
