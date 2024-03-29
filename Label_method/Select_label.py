import numpy as np
import cv2
from tkinter import *
import os
import argparse
import random


#Image inital
def Img_init():
    img_init = np.zeros((720, 1280, 3), np.uint8)
    img_init.fill(200)
    return img_init

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

#Connect N*Keppoint
def Draw_kepyoint(data,Pairs,img,No_img):
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
    step=0
    for pairA,pairB in Pairs:

        if (data[No_img][pairA][0] == 0 and data[No_img][pairA][1] == 0) or (
                data[No_img][pairB][0] == 0 and data[No_img][pairB][1] == 0):
            pass
        else:

            cv2.line(img, (int(data[No_img][pairA][0]), int(data[No_img][pairA][1])),
                     (int(data[No_img][pairB][0]), int(data[No_img][pairB][1])),color[step], 3)
            cv2.circle(img, (int(data[No_img][pairA][0]), int(data[No_img][pairA][1])),3, (0, 0, 0),  thickness=-1, lineType=cv2.FILLED)
        step += 1

    cv2.putText(img,
                str(No_img),
                (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 5)
    cv2.imshow("Connect_img", img)


#prcocess cancel Append No.img
def Cancel_Append(Append_list, content):
      try:
          if int(content.get()) in Append_list:
            Append_list.remove(int(content.get()))
            print('Cancel Append:'+content.get())
            print("AppendList:", Append_list)
          else:
              print('non-existent')
      except  ValueError :
          print("Don't input String")




#process Save Appendlist
def Save_Append(Append_list, content):
    with open('AppendList/'+content.get()+'.txt', "w") as fs:
        for i in Append_list:
            fs.write(str(i) + "\n")
    print('Successful Save '+content.get()+'.txt')

#Use Append list to process original data and Save process data
def Process_and_Save_Data(data ,Append_list,content):
    process_data=[]
    for i in Append_list:
        process_data.append(data[i])
    process_data=np.asarray(process_data)
    np.save('Process_data/'+content.get()+'.npy',process_data)
    print('Successful process and Save to '+content.get()+'.npy')


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


def Load_Processdata(content):
    global data
    if os.path.isfile('Process_data/'+content.get()+'.npy'):
        data = Load_data('Process_data/'+content.get()+'.npy')
        data = data.reshape(len(data), 25, 3)
        print('Successful Load Data')

    else:
        print('Load Failed')


#Create tkinter GUI window
def Create_window(window,label,content,btn):
    label.pack()
    content.pack()
    btn.pack()
    window.mainloop()



if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", default="Original_data/sv/sv34.npy", help="load the numpy type data")
        args = parser.parse_known_args()
        #Load Original_data ＆ Reshape data  75 to 25*3

        data= Load_data(args[0].data_path)
        #data = data.reshape(len(data), 25, 3)
        #parameter No_img is Number of img , model type is choice model
        No_img=0
        model_type=0
        #Use the corresponding posePartPairs according to model_type
        Pairs=PosePartPairs(model_type)

        #Create Append list
        Append_list=[]



        while True:

            # img initial
            img = Img_init()
            # Connect Keypoint && Show Current Img
            Draw_kepyoint(data, Pairs, img, No_img)

            # q is exit,z is next picture
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):

                if No_img < len(data)-1:
                    No_img = No_img + 1
                else:
                    print("This is last picture")

            elif key == ord("z"):
                if No_img > 0:
                    No_img = No_img - 1
                else:
                    print("This is first picture")
            elif key == ord('w'):
                Append_list=[]
                print('Clear Append list',Append_list)
            elif key == ord("s"):
                    print(" Append_list: ", Append_list)
                    print('Number of Photo:',len(Append_list))
            elif key == ord("r"):
                    Append_list.reverse()
                    print('Reverse AppendList:',Append_list)

            elif key == ord("a"):
               if No_img not in Append_list:
                    Append_list.append(No_img)
                    Append_list.sort()
                    print('Append '+str(No_img))
                    print(" Append_list: ", Append_list)
               else:
                   print(str(No_img)+' is exist')


            elif key == ord("d"):
                c_window = Tk()
                c_window.title('Cancel_Append')
                c_window.geometry('200x100')
                c_label = Label(c_window, text='Cancel Append No.Image')
                c_content = Entry(c_window, borderwidth=5)
                c_btn = Button(c_window, text='click', command=lambda: Cancel_Append(Append_list, c_content))
                Create_window(c_window,c_label,c_content,c_btn)


            elif key == ord("f"):
                s_window = Tk()
                s_window.title('Save_AppendList')
                s_window.geometry('200x100')
                s_label = Label(s_window, text='Input your Filename')
                s_content = Entry(s_window, borderwidth=5)
                s_btn = Button(s_window, text='click', command=lambda: Save_Append(Append_list, s_content))
                Create_window(s_window,s_label,s_content,s_btn)

            elif key == ord('y'):
                ps_window = Tk()
                ps_window.title('Save_ProcessData')
                ps_window.geometry('200x100')
                ps_label = Label(ps_window, text='Input your Filename')
                ps_content = Entry(ps_window, borderwidth=5)
                ps_btn = Button(ps_window, text='click', command=lambda: Process_and_Save_Data(data,Append_list, ps_content))
                Create_window(ps_window,ps_label,ps_content,ps_btn)

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

            elif key == ord('n'):

                Append_list=random.sample(range(len(data)),30)
                Append_list.sort()
                print('Random Select:',Append_list)








        #Close opencv
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)



#NOTE: 未來須優化整理的地方
#1.變數名字重新整理
#2.註解補齊
#3.function須分區塊解釋"--------"