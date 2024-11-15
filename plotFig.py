from matplotlib import pyplot as plt
import cmath
import numpy as np
from scipy.signal import butter, lfilter, freqz
import pandas as pd

#pwd:D:\StudyFiles\ILoveStudy\pro\BreatheDetect-master\data




#判断是否做相位补偿、是否滤波、是否消除0频信号,选择cir信道
ifFilter=0
cutoff=1

ifLimitTimeRange=0
time_begin,time_end=10,50

ifPhaseCompensate=1
ifMinusMean=1
elseAdd=0

ifPrintMax=1
ifDebug=0

ifSlideWindow=0
window_length=20 #窗口长度：20s  呼吸频率：0.34
window_moving_speed=1 #窗口每一秒更新一次

cir=0
filename = 'data_quiet.txt'
stepTime=0.01 #假设每个数据对应0.01s


#巴特沃斯低通滤波：设置参数
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a
##巴特沃斯低通滤波：调用函数
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#选取某段时间的数据；time_end=-1表示不限制end
def limit_time_range(data,axisTime,time_begin,time_end):
    if time_end==-1:
        data=data=data[time_begin*100:] #if (0<=time_begin<time_end and time_end<data.size()) else -1
        axisTime=axisTime[time_begin*100:]
    else:
        data=data[time_begin*100:time_end*100]
        axisTime=axisTime[time_begin*100:time_end*100]
    return data,axisTime

def read_data(filename):
    # 相比open(),with open()不用手动调用close()方法
    with open(filename, 'r') as f:
        next(f)#跳过第一行的数据
        lines = f.readlines() 
        #数据格式：第一行乱码 第二行起为：F7188 R0 1463 -749 -558 2044...长度为82；CIR共有0-39共40个信道
        amp,phase,axisTime=[],[],[]#分别记录每一行的幅值，相角，横坐标时间；幅值未使用
        tempTime=0
        linePhase,lastOriginalPhase=100,100
        flag=0 #flag=n表示当前数据在(n*pi-pi,n*pi+pi)的区间内
        for line in lines:
            lineData=line.split()#按空格分割，lineData.length=40*2+2=82
            if len(lineData)>=82:#防止最后一行数据不全
                x,y=int(lineData[2*cir+2]),int(lineData[2*cir+3])#取第n个cir信道的数据
                z = complex(x, y)
                lineAmp = abs(z)
                mutationLimit=6
                #相位补偿：用原始数据的前后两帧判断是否发生突变，用flag标示当前区间
                if ifPhaseCompensate:
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


                    
                #     elif linePhase>=0 and linePhase-cmath.phase(z)>=mutationLimit:
                #         linePhase = cmath.phase(z)
                #     elif linePhase>=0 and linePhase-cmath.phase(z)<mutationLimit:
                #         linePhase = cmath.phase(z)+cmath.pi*2
                #     elif linePhase<0 and linePhase-cmath.phase(z)<=mutationLimit:
                #         linePhase = cmath.phase(z)
                #     elif linePhase<0 and linePhase-cmath.phase(z)>mutationLimit: 
                #         linePhase = cmath.phase(z)-cmath.pi*2
                # else:
                #     linePhase = cmath.phase(z)

                amp.append(lineAmp)
                phase.append(linePhase)
                axisTime.append(tempTime)
                tempTime+=stepTime
    phase=phase[2000:4000]
    axisTime=axisTime[2000:4000]
    return phase,axisTime,amp

# magnitude = np.abs(np.fft.fft(phase)) / n
# magnitude[0] = 0
# line.set_data(frequencies[:n // 2], magnitude[:n // 2] * 2)

def minus_mean(phase):
    mean=np.mean(phase)
    phase=phase-mean+elseAdd
    return phase

# f=open("to_whx.txt","w")
# for line in phase:
#     f.write(str(line)+'\n')
# f.close()

#我们已经对相位-时间做了预处理，因此不再分析原信号的相频特性，而是分析预处理后的相位-时间图的幅频特性
def fft(phase,axisTime):
    n=len(axisTime)
    frequencies = np.fft.fftfreq(n, stepTime)
    xData=frequencies[:n // 2]
    fft=np.fft.fft(phase,n,axis=0)/n*2 #amp是幅值关于时间分布的数组
    phaseFFT=np.abs(fft)
    yData=phaseFFT[:n // 2]
    return xData,yData

#画图
def plot_fig(xData,yData):
    fig = plt.figure(figsize=(10, 10))  # 创建绘图窗口，并设置窗口大小
    # 画第一张图,相位-时间
    ax1 = fig.add_subplot(211)  # 将画面分割为2行1列选第一个
    ax1.plot(axisTime, phase, 'blue', label='phase-time')
    #ax1.set_xlim(35,40)
    ax1.set_xlabel('time(s)')  # 设置X轴名称
    ax1.set_ylabel('phase')  # 设置Y轴名称
    # 画第二张图，相频特性
    ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
    #print(len(phaseFFT))
    #ax2.plot(frequencies, np.real(phaseFFT), 'blue', label='phase-frequence')
    ax2.plot(xData, yData, 'blue', label='phase-frequence')
    ax2.set_xlim(0,5)
    ax2.set_xlabel('frequence(Hz)')  # 设置X轴名称
    ax2.set_ylabel('phase')  # 设置Y轴名称
    # ax2.plot(step, gan, 'blue', label='gan')  # 画gan-loss的值，颜色蓝
    # ax2.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
    # ax2.set_xlabel('step')
    # ax2.set_ylabel('Generator-loss')
    max_idx = np.argmax(yData)
    max_x, max_y = xData[max_idx], yData[max_idx]
    if ifPrintMax:
        plt.scatter(max_x, max_y, color='red', s=50)
        plt.annotate(f'max: ({max_x:.2f}, {max_y:.2f})', xy=(max_x, max_y),
                xytext=(max_x+0.05, max_y-0.05))

    plt.show()  # 显示绘制的图

if __name__=="__main__":
    phase,axisTime,amp=read_data(filename)

    if ifLimitTimeRange:
        phase,axisTime=limit_time_range(phase,axisTime,time_begin,time_end)
        print("len(data):",len(phase))

    if ifMinusMean:
        phase=minus_mean(phase)

    if ifFilter:
        phase=butter_lowpass_filter(phase,cutoff,fs=100)

    xData,yData=fft(phase,axisTime)
    plot_fig(xData,yData)












