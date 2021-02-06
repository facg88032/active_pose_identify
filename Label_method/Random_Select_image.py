import numpy as np
import os
from utils import Utils
import configparser
import argparse

'''
Perform N stratified random sampling of a label video clip  to enlarge the number of data
ex: 40 frames randomly selected and reduced to 30 frames * N data
'''
def main(class_name,num_sample,times,folder_path):
    #util 自行撰寫的小工具
    util=Utils()
    tmp = np.asarray([])
    for file in os.listdir(folder_path+'/'+class_name):
        #load data
        original_data=np.load(folder_path+'/'+class_name+'/'+file)
        for time in range(times):
            #Generate stratified random sample Indexs
            SampleIndexs=util.StratifiedRandomSample(len(original_data),num_sample)
            # Convert the original data into sample data by Indexs
            sample_data = util.DataProcess(original_data, SampleIndexs)
            #combine the all sample data
            tmp=util.combine(tmp,sample_data)
        print('Combine data shape:', tmp.shape)
    #drop duplicatedata
    total_data=tmp.reshape(int(len(tmp)/num_sample),num_sample*25*3)
    total_data=util.DropDuplicate(total_data)
    #save all sample data
    np.save(class_name+'.npy',total_data)
    print('total number of data',total_data.shape)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path','-f', default="Label_datasets", help="label datasets in folder")
    parser.add_argument('--num_sample','-s',default=30 ,help='num_sample number of image')
    args = parser.parse_known_args()
    folder=args[0].folder_path
    sample=args[0].num_sample
    cf = configparser.ConfigParser()
    cf.read('config.ini')
    secs = cf.sections()
    for sec in secs:
        class_name = cf.get(sec, 'class_name')
        times = cf.get(sec, 'times')
        main( class_name, sample ,int(times), folder)

