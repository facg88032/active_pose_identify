import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
data_frame=30
X_test=np.load('x_test.npy')
Y_test=np.load('y_test.npy')



X_test=X_test.reshape(len(X_test)*data_frame,25*3)
X_test=preprocessing.normalize(X_test)


blocks = int(len(X_test) / data_frame)
X_test = np.array(np.split(X_test, blocks))


mpose = keras.models.load_model('weights-improvement-99-0.99.hdf5')

output = mpose.predict_classes(X_test)

print(pd.crosstab(Y_test,output,rownames=['label'],colnames=['predict']))





