
import cv2
import time
indexs=[]
with open('shoot_log.txt','r') as fs:
    for text in fs:
        text=text.strip()
        indexs.append(text)

Video='basketball_shot480p/'+'sv1.mp4'
vs=cv2.VideoCapture(Video)
width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
max_frame=40
for index in indexs:

    vs.set(cv2.CAP_PROP_POS_FRAMES,int(index))
    # 使用 XVID 編碼
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/out'+str(index)+'.avi', fourcc, 30.0, (640, 480))
    for i in range(max_frame-1):


        #Get frame from video or webcam
        ret ,frame=vs.read()
        if not ret:
            print('no_video')
            break
        out.write(frame)
        cv2.imshow('frame',frame)

vs.release()
out.release()
cv2.destroyAllWindows()





