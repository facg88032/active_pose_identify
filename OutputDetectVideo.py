import cv2
import os

def combinIndex(indexs):
    start=0
    end=0
    max_frame=40
    combine_index=[]
    for index in range(len(indexs)) :
        if index==0:
            start=indexs[index]
            end=indexs[index]+max_frame
        elif (indexs[index]-indexs[index-1])<=3:
            end=indexs[index]+max_frame
        else:
            combine_index.append((start,end))
            start=indexs[index]
            end = indexs[index] + max_frame
        if index == len(indexs)-1:
            combine_index.append((start, end))
    return combine_index



def OutputPartVideo(indexs,vs,class_name):
    for index in indexs:
        start,end=index
        vs.set(cv2.CAP_PROP_POS_FRAMES,start)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/'+class_name+'/'+str(start)+'.avi', fourcc, 30.0, (640, 480))
        for i in range(start,end):
            ret ,frame=vs.read()
            if not ret:
                print('no_video')
                break
            out.write(frame)

    out.release()


dribble_indexs = []
shoot_indexs = []
if os.path.isfile('shoot_log.txt') and os.path.isfile('dribble_log.txt'):
    with open('shoot_log.txt','r') as s_logs:
        for s_log in s_logs:
            s_log=s_log.strip()
            shoot_indexs.append(int(s_log))
    with open('dribble_log.txt','r') as d_logs:
        for d_log in d_logs:
            d_log=d_log.strip()
            dribble_indexs.append(int(d_log))

Video='Video/5-7NewData_P/'+'ND2adm2.mp4'
vs=cv2.VideoCapture(Video)
width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)





if len(shoot_indexs)!=0:
    OutputPartVideo(combinIndex(shoot_indexs),vs,class_name='shoot')
if len(dribble_indexs)!=0:
    OutputPartVideo(combinIndex(dribble_indexs),vs,class_name='dribble')

vs.release()
cv2.destroyAllWindows()








