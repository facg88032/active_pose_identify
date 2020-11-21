import numpy as np

data1=np.load('Process_data/d3_1.npy')

data2=np.load('Process_data/d3_2.npy')

new_data=np.vstack((data1,data2))

np.save('Process_data/d3c.npy',new_data) 123