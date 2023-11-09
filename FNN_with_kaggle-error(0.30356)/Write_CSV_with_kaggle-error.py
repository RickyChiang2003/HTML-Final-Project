import cv2
import pickle
import imutils
import random
import math
import os.path
import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.models import load_model, Input    
import csv

def kaggle_error(y_true, y_pred):
    ans = 0.0
    for i in range(len(y_true)):
        ans += 3*abs(y_true[i][0]-y_pred[i][0])*(abs(y_true[i][0]-1/3)+abs(y_true[i][0]-2/3))
    ans /= float(len(y_true))
    return ans

def find_day(y, m, d):
    if m==1 or m==2:
        m += 12
        y -= 1
    return (d+2*m+3*(m+1)//5+y+y//4-y//100+y//400+1)%7

y = 2023
files = np.loadtxt('sno_test_set.txt')
bikes_number = np.load('bikes_number.npy',"r")
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'sbi'])
    m = 10
    for d in range (21, 25):
        cnt = 0
        print(cnt)
        if 1:
            for file in files:
                cnt += 1
                if cnt <= 112:
                    model = load_model("model_"+str(file)[0:9]+".hdf5",compile = False)
                    
                    for time in range(0, 1440, 20):                    
                        input_data = np.zeros(8)
                        input_data[0] = (time/60)*0.1
                        input_data[find_day(y, m, d)+1] = 1
                        input_data = np.reshape(input_data,(1,8))
                        ans = model.predict(input_data)[0][0]
                        id = str(y)
                        if(m < 10):
                            id += "0" + str(m)
                        else:
                            id += str(m)
                        id+=str(d)+"_"+str(file)[0:9]+"_"
                        if(time // 60 == 0):
                            id += "00"
                        elif(time // 60 < 10):
                            id += "0" + str(time//60)
                        else:
                            id += str(time//60)
                        id += ':'
                        if(time % 60 == 0):
                            id += "00"
                        elif(time % 60 < 10):
                            id += "0" + str(time%60)
                        else:
                            id += str(time%60)
                        writer.writerow([id, ans*bikes_number[cnt-1]])
                else:
                    for time in range(0, 1440, 20):
                        id = str(y)+str(m)+str(d)+"_"+str(file)[0:9]+"_"
                        if(time // 60 == 0):
                            id += "00"
                        elif(time // 60 < 10):
                            id += "0" + str(time//60)
                        else:
                            id += str(time//60)
                        id += ':'
                        if(time % 60 == 0):
                            id += "00"
                        elif(time % 60 < 10):
                            id += "0" + str(time%60)
                        else:
                            id += str(time%60)
                        writer.writerow([id, 1.0])
    
    m = 12
    for d in range (4, 11):
        print(cnt)
        cnt = 0
        if 1:
            for file in files:
                cnt += 1
                if cnt <= 112:
                    model = load_model("model_"+str(file)[0:9]+".hdf5",compile = False)
                    for time in range(0, 1440, 20):
                        input_data = np.zeros(8)
                        input_data[0] = (time/60)*0.1
                        input_data[find_day(y, m, d)+1] = 1
                        input_data = np.reshape(input_data,(1,8))
                        ans = model.predict(input_data)[0][0]
                        id = str(y)+str(m)
                        if(d < 10):
                            id += "0" + str(d)
                        else:
                            id += str(d)
                        id+="_"+str(file)[0:9]+"_"
                        if(time // 60 == 0):
                            id += "00"
                        elif(time // 60 < 10):
                            id += "0" + str(time//60)
                        else:
                            id += str(time//60)
                        id += ':'
                        if(time % 60 == 0):
                            id += "00"
                        elif(time % 60 < 10):
                            id += "0" + str(time%60)
                        else:
                            id += str(time%60)
                        writer.writerow([id, ans*bikes_number[cnt-1]])
                else:
                    for time in range(0, 1440, 20):
                        id = str(y)+str(m)
                        if(d < 10):
                            id += "0" + str(d)
                        else:
                            id += str(d)
                        id+="_"+str(file)[0:9]+"_"
                        if(time // 60 == 0):
                            id += "00"
                        elif(time // 60 < 10):
                            id += "0" + str(time//60)
                        else:
                            id += str(time//60)
                        id += ':'
                        if(time % 60 == 0):
                            id += "00"
                        elif(time % 60 < 10):
                            id += "0" + str(time%60)
                        else:
                            id += str(time%60)
                        writer.writerow([id, 1.0])
    
    """data = np.load("training_set_simple/"+file[0:9]+".npy")
    labels = np.load("training_set_simple/500101002_label.npy")
    model = load_model("model_500101002.hdf5")
    for i in range(1000,2000):
        input_data = data[i]
        input_data = np.reshape(input_data,(1,2))
        prediction = model.predict(input_data)
        print(prediction, end=' ')
        print(labels[i])"""