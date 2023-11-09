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

def kaggle_error(y_true, y_pred):
    ans = 0.0
    for i in range(len(y_true)):
        ans += 3*abs(y_true[i][0]-y_pred[i][0])*(abs(y_true[i][0]-1/3)+abs(y_true[i][0]-2/3))
    ans /= float(len(y_true))
    return ans

files = np.loadtxt('sno_test_set.txt')

T=0
for file in files:
    T+=1
    #if T < 1287:
    #    continue
    print(T)
    MODEL_FILENAME = "model_"+str(file)[0:9]+".hdf5"       #模型檔案名
    MODEL_LABELS_FILENAME = "model_"+str(file)[0:9]+"labels.dat"  #標籤檔案名

    data = np.load("training_set_simple/"+str(file)[0:9]+".npy")
    labels = np.load("training_set_simple/"+str(file)[0:9]+"_label.npy")
    print(len(data),len(labels))
    data2 = np.zeros((len(data), 8))
    for i in range(len(data)):
        data2[i][0] = data[i][1]*0.1
        data2[i][int(data[i][0])+1] = 1
        labels[i] = labels[i] / data[i][2]

    # 將資料分為訓練集和驗證集
    (X_train, X_test, Y_train, Y_test) = train_test_split(data2, labels, test_size=0.2, random_state=1126)

    # 建立神經網路
    model = Sequential()
    model.add(Input(shape=(8)))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(250, activation="relu"))
    model.add(Dense(75, activation="relu"))
    model.add(Dense(1, activation="linear")) # 輸出層


    # 用Keras編譯模型
    model.compile(loss=kaggle_error, optimizer="adam", metrics=[kaggle_error])

    train = 1
    if train == 1:
        # 訓練神經網路
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=3, verbose=1)

    # 儲存訓練完的模型
    model.save(MODEL_FILENAME)
    tf.saved_model.save(model, "saved_model_keras_dir")

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_keras_dir") # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
      f.write(tflite_model)


    """data = np.load("training_set_simple/"+file[0:9]+".npy")
    labels = np.load("training_set_simple/"+file[0:9]+"_label.npy")
    for i in range(0, 1440, 10):
        input_data = data[i][0:2]
        data2 = np.zeros(8)
        data2[0] = input_data[1]*0.1
        data2[int(input_data[0])+1] = 1
        input_data = np.reshape(data2,(1,8))
        prediction = model.predict(input_data)
        print(prediction, end=' ')
        print(labels[i])"""