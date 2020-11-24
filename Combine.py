import numpy as np
import os


def main(class_name):
    path='Process_data/Finish/'+class_name
    combine_data=np.asarray([])
    total_file=0
    for file in os.listdir(path):
        if combine_data.size==0:
            combine_data=np.load(path+'/'+file)
        else:
            combine_data = np.vstack((combine_data,np.load(path+'/'+file)))
        total_file+= 1
    combine_data=combine_data.reshape((total_file,30,25,3))
    np.save(class_name+'_data.npy',combine_data)
    print(combine_data.shape)

if __name__ == '__main__':
    class_name='dribble'
    main(class_name)