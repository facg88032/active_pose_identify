import numpy as np
import os
import random

def SaveAppendList(Append_list,name,time,class_name):
    filename=name.strip('.npy')+'_'+str(time)+'f'+'.txt'

    with open('AppendList/Finish/'+class_name+'/'+filename, "w") as fs:
        for i in Append_list:
            fs.write(str(i) + "\n")
    print('Successful Save '+filename)

def Processdata(data,Append_list,name,time,class_name):
    filename=name.strip('.npy')+'_'+str(time)+'f'+'.npy'
    process_data = []
    for i in Append_list:
        process_data.append(data[i])
    process_data = np.asarray(process_data)
    np.save('Process_data/Finish/'+class_name+'/' + filename, process_data)
    print('Successful process and Save to ' +filename)


def random_selsct(times,class_name):

    #load  All Files in the folder
    for file in os.listdir('Process_data/Untreated/'):
        data=np.load('Process_data/Untreated/'+file)
        #According to the specified number of times,
        # and randomly select 30 images each time,
        # and save the results of each time and generate a new data
        for time in range(times):
            Append_list = random.sample(range(len(data)), 30)
            Append_list.sort()
            SaveAppendList(Append_list,file,time+1,class_name)
            Processdata(data,Append_list,file,time+1,class_name)

            print('Random Select:', Append_list)




if __name__ == '__main__':

    class_name='shoot'
    times=30
    random_selsct(times,class_name)