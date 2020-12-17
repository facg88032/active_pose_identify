import numpy as np
import os


def main(path):
    list1=os.listdir(path)
    for i in range(len(list1)):
        a=np.load(path+'/'+list1[i])
        for j in range(i+1,len(list1)):
            b=np.load(path+'/'+list1[j])
            #check data not duplicate
            if(a==b).all():
                print(list1[i],list1[j]+' is same ')

    print('Check Finish')

if __name__ == '__main__':
    class_name='dribble'
    path='Process_data/half_'+class_name+'_e'
    main(path)
