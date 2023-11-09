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

T=0
for file in os.listdir("training_set_simple"):
    if(file[-5] == 'l'):
        continue
    T+=1
    if T < 1287:
        continue
    print(T)
    MODEL_FILENAME = "model_"+file[0:9]+".hdf5"       #模型檔案名
    MODEL_LABELS_FILENAME = "model_"+file[0:9]+"labels.dat"  #標籤檔案名

    data = np.load("training_set_simple/"+file[0:9]+".npy")
    labels = np.load("training_set_simple/"+file[0:9]+"_label.npy")
    print(len(data),len(labels))

    # 將資料分為訓練集和驗證集
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=666)

    # 建立神經網路
    model = Sequential()
    model.add(Input(shape=(2)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(1, activation="linear")) # 輸出層


    # 用Keras編譯模型
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

    train = 1
    if train == 1:
        # 訓練神經網路
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=5, verbose=1)

    # 儲存訓練完的模型
    model.save(MODEL_FILENAME)
    tf.saved_model.save(model, "saved_model_keras_dir")

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_keras_dir") # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
      f.write(tflite_model)


    """data = np.load("training_set_simple/500101001.npy")
    labels = np.load("training_set_simple/500101001_label.npy")
    for i in range(1000):
        input_data = data[i]
        input_data = np.reshape(input_data,(1,2))
        prediction = model.predict(input_data)
        print(prediction, end=' ')
        print(labels[i])"""
