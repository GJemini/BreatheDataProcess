import numpy as np
import matplotlib.pyplot as plt
import threading
import time

X=0
Y=0

lock=threading.Lock()

def addOne():
    global X
    while X<=10:
        with lock:
            X=X+1
            print("X=",X," ")
        time.sleep(1)
        
    

def minusOne():
    global Y
    while Y>=-10:
        with lock:
            Y=Y-1
            print("Y=",Y," ")
        time.sleep(1)
        
def plot_thread(x,y):
    plt.ion()

    

if __name__=="__main__":
    thread1 = threading.Thread(target=addOne)
    thread2 = threading.Thread(target=minusOne)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
