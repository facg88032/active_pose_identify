import numpy as np
import cv2
from tkinter import *
import os
import argparse
import time



#Load data
def Load_data(data_name):
    data=np.load(data_name)
    return data

#Different models have different posePartPairs
def  PosePartPairs(model_type):
    #0 is Body_25 ,1 is COCO
    if model_type == 0:
        Pairs=[1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21, 11,22,22,23,11,24]
        Pairs=np.asarray(Pairs).reshape(24,2)
    else :
        Pairs=[1,2,3,4,3,2,1,5,6,7,6,5,1,0,14,16,14,0,15,17,15,0,1,8,9,10,9,8,1,11,12,13,12,11,1]
    return Pairs

#Image inital
def Img_init():
    img_init = np.zeros((600, 960, 3), np.uint8)
    img_init.fill(200)
    return img_init

#Connect N*Keppoint
def Draw_kepyoint(data,Pairs,img,No_img,current):
    color = [(0, 0, 255), #1 8
             (0, 85, 255),#1 2
             (0, 255, 170),#1 5
             (0, 170, 255),#2 3
             (0, 255, 255),#3 4
             (0, 170, 0),#5 6
             (0,100,0),#6 7
             (85, 170, 0),#8 9
             (140, 170, 0),#9 10
             (255, 255, 0),#10 11
             (255, 170, 0),#8 12
             ( 255, 85,0),#12 13
             (255, 0, 0),#13 14
             (85,0, 255),#1 0
             (170, 0, 255),# 0 15
             (85, 0, 255),#15 17
             (255, 0, 170),#0 16
             (255, 0, 85),#16 18
             (255, 0, 0),#14 19
             (255, 0, 0),#19 20
             (255, 0, 0),#14 21
             (255, 255, 0),#11 22
             (255, 255, 0),#22 23
             (255, 255, 0),]#11 24
    color_id=0
    for pairA,pairB in Pairs:

        if (data[No_img][pairA][0] == 0 and data[No_img][pairA][1] == 0) or (
                data[No_img][pairB][0] == 0 and data[No_img][pairB][1] == 0):
            pass
        else:

            cv2.line(img, (int(data[No_img][pairA][0]), int(data[No_img][pairA][1])),
                     (int(data[No_img][pairB][0]), int(data[No_img][pairB][1])),color[color_id], 2)
            cv2.circle(img, (int(data[No_img][pairA][0]), int(data[No_img][pairA][1])),3, (0, 0, 0),  thickness=-1, lineType=cv2.FILLED)
        color_id += 1

    cv2.putText(img,
                str(current),
                (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 5)
    cv2.putText(img,
                str(No_img),
                (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 5)
    cv2.imshow("Connect_img", img)


#load Append_list
def Load_list(content):
    global Append_list
    Append_list=[]
    if os.path.isfile('AppendList/'+content.get()+'.txt'):
        with open('AppendList/'+content.get() + '.txt', "r") as fl:
            for i in fl:
                i = int(i.strip())
                Append_list.append(i)
        print('Successful Load Append List')
        print(" Append_list: ", Append_list)
    else:
        print('Load Failed')

#load Processdata
def Load_Processdata(content):
    global data
    if os.path.isfile('Process_data/'+content.get()+'.npy'):
        data = Load_data('Process_data/'+content.get()+'.npy')
        data = data.reshape(len(data), 25, 3)
        print('Successful Load Data')

    else:
        print('Load Failed')

#Use Append list to process original data
#Save process data and Append_list
def Process_and_Save_Data(data ,Append_list,data_name,class_name,label_number):
    filename=data_name+'_'+class_name[0]+'_'+str(label_number)+'e'
    process_data=[]
    for i in Append_list:
        process_data.append(data[i])
    process_data=np.asarray(process_data)
    with open('AppendList/half_'+class_name+'/'+filename+'.txt', "w") as fs:
        for i in Append_list:
            fs.write(str(i) + "\n")
    np.save('Process_data/half_'+class_name+'/'+ filename+ '.npy', process_data)
    print('Successful process and Save to '+filename+'.npy')

#Create tkinter GUI window
def Create_window(window,label,content,btn):
    label.pack()
    content.pack()
    btn.pack()
    window.mainloop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",'-d', default="Original_data/shoot/s5.npy", help="load the numpy type data")
    args = parser.parse_known_args()
    #Load Original_data ＆ Reshape data  75 to 25*3
    # parameter No_img is Number of img , model type is choice model
    # Use the corresponding posePartPairs according to model_type
    # Create Append list
    No_img=0
    model_type = 0
    current= 0
    step = 0
    max_frame=40
    move_space= 1
    LabelNo_d = 1
    LabelNo_s = 1
    LabelNo_o = 1
    Append_list = [i for i in range(max_frame)]
    Pairs=PosePartPairs(model_type)
    data = Load_data(args[0].data_path)
    data_name= args[0].data_path.split("/")[-1].replace('.npy','')


    while True:

        # img initial
        # Connect Keypoint && Show Current Img
        img = Img_init()
        Draw_kepyoint(data, Pairs, img, No_img,current)
        No_img+=1
        step+=1
        time.sleep(0.05)
        if step%max_frame==0:
            No_img=current



        #  q is exit,z is next picture
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
             break

        elif key == ord("c"):
            if current < len(data)-max_frame-1:
                current = current + move_space
                Append_list=Append_list[move_space:]
                for i in range(move_space):
                    Append_list.append(Append_list[-1]+1)
            else:
                print("This is last picture")

        elif key == ord("z"):
            if current > 0:
                current = current - move_space
                Append_list = Append_list[:len(Append_list)-move_space]
                for i in range(move_space):
                    Append_list.append(Append_list[0]-1)
                    Append_list.sort()
            else:
                print("This is first picture")

        elif key == ord("w"):
                print(" Append_list: ", Append_list)
                print('dribble:',LabelNo_d-1,' shoot:',LabelNo_s-1,' other:',LabelNo_o-1)


        elif key == ord("d"):
            if len(Append_list)==max_frame:
                class_name='dribble'
                Process_and_Save_Data(data, Append_list, data_name, class_name,LabelNo_d)
                LabelNo_d+=1
                print('Current No_img:',current)
            else:
                print('Append list not enough ',max_frame,' frame')

        elif key == ord("s"):
            if len(Append_list) == max_frame:
                class_name='shoot'
                Process_and_Save_Data(data, Append_list, data_name, class_name, LabelNo_s)
                LabelNo_s += 1
                print('Current No_img:', current)
            else:
                print('Append list not enough ', max_frame, ' frame')

        elif key == ord("a"):
            if len(Append_list) == max_frame:
                class_name='other'
                Process_and_Save_Data(data, Append_list, data_name, class_name, LabelNo_o)
                LabelNo_o += 1
                print('Current No_img:', current)
            else:
                print('Append list not enough ', max_frame, ' frame')


        elif key == ord('j'):
            ld_window = Tk()
            ld_window.title('Load_AppendList')
            ld_window.geometry('200x100')
            ld_label= Label(ld_window, text='Input your Filename')
            ld_content = Entry(ld_window, borderwidth=5)
            ld_btn = Button(ld_window, text='click', command=lambda :Load_list(ld_content))
            Create_window(ld_window,ld_label,ld_content,ld_btn)

        elif key == ord('k'):
            lpd_window = Tk()
            lpd_window.title('Load_ProcessData')
            lpd_window.geometry('200x100')
            lpd_label = Label(lpd_window, text='Input your Filename')
            lpd_content = Entry(lpd_window, borderwidth=5)
            lpd_btn = Button(lpd_window, text='click', command=lambda: Load_Processdata(lpd_content))
            Create_window(lpd_window, lpd_label, lpd_content, lpd_btn)
            No_img=0

        elif key == ord('h'):
            print('Total frame:',len(data))



    #Close opencv
    cv2.destroyAllWindows()




#NOTE: 未來須優化整理的地方
#1.變數名字重新整理
#2.註解補齊
#3.function須分區塊解釋"--------"