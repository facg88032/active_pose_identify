import numpy as np




def Stack(save_array,data):

    if save_array.size !=0:
        array = np.vstack((save_array, data))
    else:
        array = data
    return array


def run(threshold,keypoints_data,select_model=0):
    each_data=np.array([])
    total_data=np.array([])

    for i  in range(len(keypoints_data)):
        each_data=Stack(each_data,keypoints_data[i])
        if each_data.shape[0] == threshold:
            total_data=Stack(total_data,each_data.reshape(1, threshold, keypoints_data.shape[1]))
            each_data=np.array([])
    if select_model==0:
        np.save('processdata.npy',total_data)
    else:
        return total_data

# Create Dict save  all data
keypoints_data = np.load('data.npy')

#N_th data , each of data have threshold_value  frame
threshold=32

run(threshold, keypoints_data)