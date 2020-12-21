import os
import shutil
import numpy as np
import argparse

def main(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            data_path=folder_path+'/'+filename
            data=np.load(data_path)
            if len(data)>=40:
                shutil.move(data_path,folder_path+'/'+'40up')
            elif len(data)>=35:
                shutil.move(data_path, folder_path + '/' + '35')
            else:
                shutil.move(data_path, folder_path + '/' + '30')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", default='dribble', help="load the class name")
    args= parser.parse_known_args()
    class_name=args[0].class_name
    folder_path= class_name

    main(folder_path)


