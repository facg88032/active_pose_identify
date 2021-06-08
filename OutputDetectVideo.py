import cv2
import os



def OutputPartVideo(indexs,vs,class_name):
    for index in indexs:
        vs.set(cv2.CAP_PROP_POS_FRAMES,int(index))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/'+class_name+'/'+str(index)+'.avi', fourcc, 30.0, (640, 480))
        for i in range(max_frame):
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
            shoot_indexs.append(s_log)
    with open('dribble_log.txt','r') as d_logs:
        for d_log in d_logs:
            d_log=d_log.strip()
            dribble_indexs.append(d_log)

Video='Video/5-7NewData_P/'+'ND2adm2.mp4'
vs=cv2.VideoCapture(Video)
width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
max_frame=40

if len(shoot_indexs)!=0:
    OutputPartVideo(shoot_indexs,vs,class_name='shoot')
if len(dribble_indexs)!=0:
    OutputPartVideo(dribble_indexs,vs,class_name='dribble')

vs.release()
cv2.destroyAllWindows()








