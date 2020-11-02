import numpy as np
import cv2



#X_train=np.loadtxt('X_train.txt',delimiter=',')
#X_train=X_train.reshape(len(X_train),18,2)
data=np.load('data.npy')
data=np.load('processdata.npy')
#data=X_train
print(data.shape)
data=data.reshape(len(data),25,3)
#ct_seq=[1,2,3,4,3,2,1,5,6,7,6,5,1,0,16,18,16,0,15,17,15,0,1,8,9,10,11,24,11,22,23,22,11,10,9,8
#        ,12,13,14,21,14,19,20]
ct_seq=[1,2,3,4,3,2,1,5,6,7,6,5,1,0,14,16,14,0,15,17,15,0,1,8,9,10,9,8,1,11,12,13,12,11,1]

img = np.zeros((720,1280, 3), np.uint8)
img.fill(200)
#Number of img
No_img=0
count=No_img

while True:


    cv2.imshow("data1", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("z"):
        if No_img < len(data):
            img = np.zeros((720, 1280, 3), np.uint8)
            img.fill(200)
            for i in range(len(ct_seq) - 1):
                if (data[No_img][ct_seq[i]][0]==0 and data[No_img][ct_seq[i]][1]==0) or (data[No_img][ct_seq[i + 1]][0] ==0 and data[No_img][ct_seq[i + 1]][1]==0):
                    pass
                else:
                    print(data[No_img][ct_seq[i]][0])
                    cv2.line(img, (int(data[No_img][ct_seq[i]][0]), int(data[No_img][ct_seq[i]][1])),
                             (int(data[No_img][ct_seq[i + 1]][0]), int(data[No_img][ct_seq[i + 1]][1])), (0, 0, 0), 5)

            cv2.putText(img,
                        str(No_img+1),
                        (500,500), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 0), 5)
            cv2.imshow("catch", img)

            No_img=No_img+1
        else:
            print("This is last picture")

cv2.destroyAllWindows()