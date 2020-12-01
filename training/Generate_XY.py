import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



dribble=np.load('dribble_data.npy')
shoot=np.load('shoot_data.npy')
other=np.load('other_data.npy')
#make label and merge label
labels = np.zeros(len(dribble))
labels = np.append(labels, np.full((len(shoot)), 1))
labels = np.append(labels, np.full((len(other)), 2))

#merge data
dataset=np.vstack((dribble,shoot))
dataset=np.vstack(((dataset,other)))

#shuffle all data
x_data, y_data = shuffle(dataset, labels)


#resize x_data
x_data[:,:,:,0] = x_data[:,:,:,0] / 480
x_data[:,:,:,1] = x_data[:,:,:,1] / 720

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33)

np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)

print('Generated')