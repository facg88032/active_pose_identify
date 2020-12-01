# Import the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.optimizers import Adam ,SGD
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np




X_train=np.load('x_train.npy')
Y_train=np.load('y_train.npy').reshape(-1,1)
enc=OneHotEncoder()
Y_train=enc.fit_transform(Y_train).toarray()


X_train=X_train.reshape((len(X_train,)*30,25*3))
X_train=pd.DataFrame(X_train)




def normalize(train):

  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

X_train=normalize(X_train)
X_train=X_train.values
blocks = int(len(X_train) / 30)
X_train = np.array(np.split(X_train, blocks))

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units =50,return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Activation('relu'))
regressor.add(Dropout(0.5))


for i in range(2):
    regressor.add(LSTM(units = 50,return_sequences=True))
    regressor.add(Activation('relu'))
    regressor.add(Dropout(0.5))


# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Activation('relu'))
regressor.add(Dropout(0.5))


# Adding the output layer
regressor.add(Dense(3,activation='softmax'))
learning_rate = 0.1
decay_rate = learning_rate / 100000
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# Compiling
regressor.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]
regressor.summary()
# 進行訓練
regressor.fit(X_train, Y_train, validation_split=0.3,epochs =50,callbacks=callbacks_list,batch_size = 200)