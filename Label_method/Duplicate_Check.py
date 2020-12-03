import numpy as np
import os


def main(class_name):
    list1=os.listdir('Process_data/Finish/'+class_name)
    for i in range(len(list1)):
        a=np.load('Process_data/Finish/'+class_name+'/'+list1[i])
        for j in range(i+1,len(list1)):
            b=np.load('Process_data/Finish/'+class_name+'/'+list1[j])
            #check data not duplicate
            if(a==b).all():
                print(list1[i],list1[j]+' is same ')
        print(i)
    print('Check Finish')

if __name__ == '__main__':
    class_name='dribble'
    main(class_name)
