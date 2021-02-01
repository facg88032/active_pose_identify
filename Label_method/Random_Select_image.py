import numpy as np
import os
from utils import Utils
import configparser
import argparse

def main(class_name,num_select,times,folder_path):

    util=Utils()
    tmp = np.asarray([])
    #load  All Files in the folder
    for file in os.listdir(folder_path+'/'+class_name):
        original_data=np.load(folder_path+'/'+class_name+'/'+file)
        for time in range(times):
            SampleIndexs=util.StraifiedRandomSample(len(original_data),num_select)
            process_data = util.DataProcess(original_data, SampleIndexs)
            tmp=util.combine(tmp,process_data)

        print('Combine data shape:', tmp.shape)
    total_data=tmp.reshape(int(len(tmp)/num_select),num_select,25,3)
    np.save(class_name+'.npy',total_data)
    print('total number of data',len(total_data))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path','-f', default="Label_datasets", help="label datasets in folder")
    parser.add_argument('--num_select','-s',default=30 ,help='select number of image')
    args = parser.parse_known_args()
    folder=args[0].folder_path
    select=args[0].num_select
    cf = configparser.ConfigParser()
    cf.read('config.ini')
    secs = cf.sections()


    for sec in secs:
        class_name = cf.get(sec, 'class_name')
        times = cf.get(sec, 'times')
        main( class_name, select ,int(times), folder)

