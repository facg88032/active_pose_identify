import numpy as np
import os


def main():
    list1=os.listdir('Process_data/Finish')
    for i in range(len(list1)):
        a=np.load('Process_data/Finish/'+list1[i])
        for j in range(i+1,len(list1)):
            b=np.load('Process_data/Finish/'+list1[j])
            #check data not duplicate
            if(a==b).all():
                print(list1[i],list1[j]+' is same ')


if __name__ == '__main__':
    main()
