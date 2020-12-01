import cv2
from openpose import pyopenpose as op
import collections
import numpy as np
import math
vs=cv2.VideoCapture(0)

width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create objects to process pictures
datum = op.Datum()

# Create Dict save  all data
all_data={}

# Creat empty araay  save a segment of video ,each segment have N frame
each_data=np.array([])
#N_th data , each of data have threshold_value  frame
N_th=0
threshold=32

min_path=int(math.pow(540,2)+math.pow(360,2))

mid_X = 0
mid_Y = 0

while (vs.isOpened()):
    ret ,frame=vs.read()

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:

        for i  in range(len(datum.poseKeypoints)):
            X=0
            Y=0
            count = 0
            for j in range(25):
                if datum.poseKeypoints[i][j][0]!=0 or datum.poseKeypoints[i][j][1] !=0:
                    X=X+datum.poseKeypoints[i][j][0]
                    Y=Y+datum.poseKeypoints[i][j][1]
                    count+=1
            a_X=X/count
            a_Y=Y/count
            length=math.pow(width/2-a_X,2)+math.pow(height/2-a_Y,2)
            if length<min_path:
                min_path=length
                mid_keypoints=datum.poseKeypoints[i]
                mid_X=a_X
                mid_Y=a_Y







        mid_keypoints=mid_keypoints.reshape(1,25,3)
        if each_data.size !=0:
            each_data = np.vstack((each_data, mid_keypoints))
        else:
            each_data = mid_keypoints

        all_data[N_th] = each_data

        if each_data.shape[0] == threshold:
            # print(each_data.shape[0])
            # all_data[N_th] = all_data[N_th][0:threshold]
            N_th += 1
            #clean array
            each_data = np.array([])

    image = datum.cvOutputData
    cv2.putText(image,
                'mid',
                (int(mid_X), int(mid_Y)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (200, 150, 70), 5)
    cv2.imshow("Openpose", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i in all_data:
    print('Key'+str(i)+':',all_data[i].shape)

    #np.save('data'+str(i)+'.npy',all_data[i])

vs.release()
cv2.destroyAllWindows()

