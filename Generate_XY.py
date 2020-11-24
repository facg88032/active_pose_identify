import numpy as np
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical


dribble=np.load('dribble_data.npy')
shoot=np.load('shoot_data.npy')

#make label and merge label
labels = np.zeros(len(dribble))
labels = np.append(labels, np.full((len(shoot)), 1))

#merge data
dataset=np.vstack((dribble,shoot))

#shuffle all data
x_data, y_data = shuffle(dataset, labels)

#one-hot enconding
y_data= to_categorical(y_data, 2)

#resize x_data
x_data[:,:,:,0] = x_data[:,:,:,0] / 480
x_data[:,:,:,1] = x_data[:,:,:,1] / 720

np.save('x_data.npy',x_data)
np.save('y_data.npy',y_data)

print('Generated')