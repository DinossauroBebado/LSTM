import pandas as pd
# import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from collections import deque
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from rede_neural_vis import ploter
from math import sqrt

DIAS_PASSADO = 20
DIAS_FUTURO = 3
EPOCHS = 30
BATCH = 64


df_train = pd.read_csv('info_COVID.csv',
                       delimiter=';', header=0, index_col=0)  # coloca

df_train
last_20pct = df_train.index[-126]

validation_df = df_train[last_20pct:]  # pega o final para realizar o teste
main_df = df_train[:last_20pct]


print(main_df)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(main_df)
print(scaled_train)

[[1, 2, 3, 4, 5], [1, 2, 3]]


model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization)

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization)

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization)

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(Lr=0.001, decay=1e-6)

model.compile(Loss="sparce_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

history = model.fit(train_x, train_y,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    validation_data=(valid_x, valid_y))

model.save(model.save('Neocovid_predict_full.h5'))

predictions = model.predict(valid_x)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(valid_y.reshape(-1, 1))


rmse = sqrt(mean_squared_error(y_test_scaled, predictions))

ploter(y_test_scaled, predictions, EPOCHS, BATCH, DIAS_PASSADO, rmse)'''
