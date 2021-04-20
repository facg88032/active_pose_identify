import keras
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
data_frame=30
X_test=np.load('x_test.npy')
Y_test=np.load('y_test.npy')



X_test=X_test.reshape(len(X_test)*data_frame,25*3)
X_test=preprocessing.normalize(X_test)


blocks = int(len(X_test) / data_frame)
X_test = np.array(np.split(X_test, blocks))


mpose = keras.models.load_model('weights-improvement-98-0.99.hdf5')

output = mpose.predict_classes(X_test)

# print(pd.crosstab(Y_test,output,rownames=['label'],colnames=['predict']))

LABELS=[
    "dribble",
    "shoot",
    "other"
]

print("")
print("Confusion Matrix:")
print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(Y_test)))
confusion_matrix = metrics.confusion_matrix(Y_test, output)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

# Plot Results:
width = 10
height = 10
plt.figure()

plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.Blues
)

for x in range(normalised_confusion_matrix.shape[0]):
    for y in range(normalised_confusion_matrix.shape[1]):
        plt.text(s=str( '%.1f' % normalised_confusion_matrix[x][y])+"%",x=y,y=x,  ha='center', va= 'center',fontsize=13,color='tomato')
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, LABELS)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()





