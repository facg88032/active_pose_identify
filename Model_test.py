import keras
import numpy as np
import pandas as pd

X_test=pd.read_csv('X_test.txt', sep=",", header=None)
Y_test=np.loadtxt('Y_test.txt')

Y_test=Y_test[:5308]
# enc=OneHotEncoder()
# Y_train=enc.fit_transform(Y_train).toarray()
def normalize(train):

  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm
X_test=normalize(X_test)
X_test=X_test.values

blocks = int(len(X_test) / 32)
X_test = np.array(np.split(X_test, blocks))
X_test=X_test[:5308]

mpose = keras.models.load_model('weights-improvement-139-0.75.hdf5')

output = mpose.predict_classes(X_test)


print(pd.crosstab(Y_test,output,rownames=['label'],colnames=['predict']))





