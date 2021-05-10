# Import the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.optimizers import Adam ,SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint , TensorBoard
import pandas as pd
import numpy as np
import joblib as jb

data_frame=30

X_train=np.load('x_train.npy')
Y_train=np.load('y_train.npy').reshape(-1,1)
enc=OneHotEncoder()
Y_train=enc.fit_transform(Y_train).toarray()


X_train=X_train.reshape((len(X_train)*data_frame,25*3))
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
jb.dump(scaler,'std_scaleND.bin',compress=True)
X_train =X_train.reshape(int(len(X_train)/data_frame),data_frame,25*3)

# X_train=X_train.reshape((len(X_train)*data_frame,25*3))
# X_train=preprocessing.normalize(X_train)
# blocks = int(len(X_train) / (data_frame))
# X_train = np.array(np.split(X_train, blocks))

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

learning_rate = 0.01
decay_rate = learning_rate / 10000
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Compiling
regressor.compile(optimizer = adam , loss = 'categorical_crossentropy', metrics=['accuracy'])

#checkpoint
filepath="C_training_model/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')

#monitor
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
callbacks_list = [checkpoint ,tbCallBack]
regressor.summary()

# 進行訓練
regressor.fit(X_train, Y_train, validation_split=0.3,epochs =250,callbacks=callbacks_list,batch_size = 200)