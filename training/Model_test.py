import keras
import numpy as np
import pandas as pd

X_test=np.load('x_train.npy')
Y_test=np.load('y_train.npy')

X_test=X_test.reshape((len(X_test)*40,25*3))
X_test=pd.DataFrame(X_test)


def normalize(train):

  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

X_test=normalize(X_test)
X_test=X_test.values

blocks = int(len(X_test) / 40)
X_test = np.array(np.split(X_test, blocks))


mpose = keras.models.load_model('weights-improvement-97-0.99.hdf5')

output = mpose.predict_classes(X_test)
c=0
for i in output:
  if Y_test[c] == 2:
    print('---------------------------')
    ttt=X_test[c]
    ggg=Y_test[c]
  c+=1
print(ggg)
print(pd.crosstab(Y_test,output,rownames=['label'],colnames=['predict']))





