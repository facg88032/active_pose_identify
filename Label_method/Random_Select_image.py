import numpy as np
import os
from utils import Utils


def main(times,class_name,num_select):
    util=Utils()
    tatoal_data = np.asarray([])
    #load  All Files in the folder
    for file in os.listdir('Process_data/half_'+class_name+'/'):
        data=np.load('Process_data/half_'+class_name+'/'+file)
        #According to the specified number of times,
        # and randomly select 30 images each time,
        # and save the results of each time and generate a new data
        for time in range(times):
            SampleIndexs=util.StraifiedRandomSample(len(data),num_select)
            process_data = util.DataProcess(data, SampleIndexs)
            tatoal_data=util.combine(tatoal_data,process_data)
            print('Combine data shape:', tatoal_data.shape)
    tatoal_data=tatoal_data.reshape(int(len(tatoal_data)/num_select),num_select,25,3)
    np.save(class_name+'.npy',tatoal_data)
    print('total number of data',len(tatoal_data))

if __name__ == '__main__':

    class_name='dribble'
    times=6
    select=30
    main(times,class_name,select)