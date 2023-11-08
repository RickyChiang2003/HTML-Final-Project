import numpy as np
import csv
import os


name1 = ""
name2 = ""
totalfile = 0
for subname in range(1001, 19092):  # for each station
    if subname < 10000:
        name1 = "5001" + "0" + str(subname) + ".npy"
        name2 = "5001" + "0" + str(subname) + "_label.npy"
    else:
        name1 = "5001" + str(subname) + ".npy"
        name2 = "5001" + str(subname) + "_label.npy"
    if(os.path.isfile(name1) == True):
        totalfile += 1

csvarr = np.zeros([totalfile, 8, 72])
namearr = np.zeros(totalfile)
csvcnt = 0
for subname in range(1001, 19092):  # for each station
    if subname < 10000:
        name1 = "5001" + "0" + str(subname) + ".npy"
        name2 = "5001" + "0" + str(subname) + "_label.npy"
    else:
        name1 = "5001" + str(subname) + ".npy"
        name2 = "5001" + str(subname) + "_label.npy"
    if(os.path.isfile(name1) == False):
        continue
    print("subname = ",subname)
    namearr[csvcnt] = subname
    arr1 = np.load(name1)
    arr2 = np.load(name2)

    bike = np.zeros([8,72])
    daynum = np.zeros([8,72])  
    flag = 0
    for week in range(3):
        for day in range(7):
            if (arr1[flag][0] != (day+1)%7):
                day = (int)(arr1[flag][0]+7-1)%7
            for min20 in range(72):
                if(day+1 == 6):
                    print("min20=",min20,"arr2[flag]=",arr2[flag])
                for minute in range(20):
                    if(arr1[flag][1] <= 0.01667*(min20*20+minute) and arr1[flag][0] == (day+1)%7):
                        bike[day+1][min20] += arr2[flag]
                        flag += 1
                        daynum[day+1][min20] += 0.05
    for day in range(7):
        for min20 in range(72):
            bike[day+1][min20] /= daynum[day+1][min20]*20
            csvarr[csvcnt][day+1][min20] = (int)(bike[day+1][min20])
    csvcnt += 1
#print(csvarr[0])


#csv
dayarr = [6,7,1,2]
datearr = ["21","22","23","24"]
minarr = ["00","20","40"]
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for dayid in range(4):
        for id in range(totalfile):
            #print("day=10/",datearr[dayid],", id=",id)
            for min20 in range(72):
                s = "202310"+datearr[dayid]+"_5001"
                if(namearr[id] < 10000):
                    s = s+"0"
                s = s+str((int)(namearr[id]))+"_"
                hour = (int)(min20/3)
                minute = min20%3
                if(hour < 10):
                    s = s+"0"
                s = s+str(hour)+":"+minarr[minute]
                writer.writerow([s,csvarr[id][dayarr[dayid]][min20]])
