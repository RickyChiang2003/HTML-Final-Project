import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import csv
import os

def kaggle_error(y_true, y_pred):
    ans = 0.0
    for i in range(len(y_true)):
        ans += 3*abs(y_true[i][0]-y_pred[i][0])*(abs(y_true[i][0]-1/3)+abs(y_true[i][0]-2/3))
    ans /= float(len(y_true))
    return ans

def compute_file_amount():
    totalfile = 0
    name1 = ""
    name2 = ""
    for subname in range(1001, 19092):  # for each station 1001, 19092
        if subname < 10000:
            name1 = "5001" + "0" + str(subname) + ".npy"
            name2 = "5001" + "0" + str(subname) + "_label.npy"
        else:
            name1 = "5001" + str(subname) + ".npy"
            name2 = "5001" + str(subname) + "_label.npy"
        if(os.path.isfile(name1) == True) and (os.path.isfile(name2) == True):
            totalfile += 1
    return totalfile

def init_csvarr(csvarr, totalfile):
    for i in range(totalfile):
        for j in range(8):
            for k in range(72):
                csvarr[i][j][k] = 1

def init_x_y(arr1, arr2, len1, x,y):
    for i in range(len1):
        x[i][0] = arr1[i][1]
        day = (int)(arr1[i][0])
        if day == 0:
            day = 7
        x[i][day] = 1
        x[i][8] = (1 if x[i][6] == 1 or x[i][7] == 1 or i == 7 or i == 8 else 0)
        y[i] = arr2[i]/arr1[i][2]

def init_x_y_train(x_train, y_train, n_steps, train_len):
    for i in range(n_steps,train_len):
        for j in range(i-n_steps, i):
            x_train[i-n_steps][j-i+n_steps] = x[j]
        y_train[i-n_steps] = y[i]

def init_predict_x(predict_x):
    for i in range(1,8):
        for j in range(72):
            t_x = np.zeros(9)
            t_x[0] = j*20*0.24/60
            t_x[i] = 1
            t_x[8] = (1 if i == 6 or i == 7 else 0)  # I ignore holiday after 11/20 here :)
            for k in range(n_steps):
                predict_x[((i-1)*72+j+1+k)%504][n_steps-1-k] = t_x

def output_to_csv(arrtest, totalfile, namearr, csvarr):
    dayarr = [6,7,1,2,4,5,6,7,1,2,3]
    datearr = ["1021","1022","1023","1024","1211","1212","1213","1214","1215","1216","1217"]
    minarr = ["00","20","40"]
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","sbi"])
        for dayid in range(11):
            for id in range(totalfile):
                found = False
                for i in arrtest:
                    stest = str((int)(i-500100000))
                    if stest == str((int)(namearr[id])):
                        found = True
                        break
                if found == False:
                    continue
                #print("day=10/",datearr[dayid],", id=",id)
                for min20 in range(72):
                    s = "2023"+datearr[dayid]+"_5001"
                    if(namearr[id] < 10000):
                        s = s+"0"
                    s = s+str((int)(namearr[id]))+"_"
                    hour = (int)(min20/3)
                    minute = min20%3
                    if(hour < 10):
                        s = s+"0"
                    s = s+str(hour)+":"+minarr[minute]
                    writer.writerow([s,csvarr[id][dayarr[dayid]][min20]])    



# main
arrtest = np.loadtxt("sno_test_set.txt")
totalfile = compute_file_amount()

csvarr = np.zeros([totalfile, 8, 72])  # file amount: totalfile, Monday to Sunday: [1~7], 20minutes: min20
namearr = np.zeros(totalfile)
init_csvarr(csvarr, totalfile)

csvcnt = 0
for subname in range(1001, 19092):  # for each station 1001, 19092
    name1 = ""
    name2 = ""
    if subname < 10000:
        name1 = "5001" + "0" + str(subname) + ".npy"
        name2 = "5001" + "0" + str(subname) + "_label.npy"
    else:
        name1 = "5001" + str(subname) + ".npy"
        name2 = "5001" + str(subname) + "_label.npy"
    if (os.path.isfile(name1) and os.path.isfile(name2)) == False:
        continue
    print("exist subname = ",subname)

    # pre-process
    namearr[csvcnt] = subname
    arr1 = np.load(name1)
    arr2 = np.load(name2)
    len1 = len(arr1) #56775 = 757 * 75
    len2 = len(arr2)
    x = np.zeros([len1,9])  # hour(per minute)/10, Monday to Sunday(1~7), holiday(10/09, 10/10, [6,7])
    y = np.zeros(len1)
    init_x_y(arr1, arr2, len1, x,y)
    
    # Adding LSTM layer
    n_steps = 75  # 
    n_features = 9
    model = Sequential()
    model.add(LSTM(256,activation='relu',return_sequences=False,input_shape=(n_steps,n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss=kaggle_error,metrics=[kaggle_error])
    
    # train
    train_len = 12000
    x_train = np.zeros([train_len-n_steps,n_steps,n_features])
    y_train = np.zeros([train_len-n_steps])
    init_x_y_train(x_train, y_train, n_steps, train_len)
    history = model.fit(x_train,y_train,batch_size=64,epochs=1) # can be change like 64,50
 
    # predict
    predict_x = np.zeros([504, n_steps, n_features])  # there are 504 20minutes in one week
    init_predict_x(n_steps, predict_x)
    result = model.predict(predict_x)
    # re-predict if the result is a piace of shit
    repeat_time = 0
    for i in range(len(result)):
        if result[i] > 1 or result[i] < 0:
            history = model.fit(x_train,y_train,batch_size=64,epochs=1) # can be change like 64,50
            result = model.predict(predict_x)
            repeat_time += 1
            i = 0
        elif np.isnan(result[i]) or repeat_time >= 5:
            for tmptmp in result:
                tmptmp = 1

    # store result
    id_r = 0
    for i in range(1,8):
        for j in range(72):
            csvarr[csvcnt][i][j] = arr1[2][2] * result[id_r]
            id_r += 1
    csvcnt += 1

#print(csvarr[0])


#testcsv
output_to_csv(arrtest, totalfile, namearr, csvarr)



