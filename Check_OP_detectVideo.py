import cv2
from openpose import pyopenpose as op
import numpy as np
import time
import os

Video='basketball_shot480p/'+'sv57.mp4'

vs=cv2.VideoCapture(Video)
index=2545
width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
vs.set(cv2.CAP_PROP_POS_FRAMES, index)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["model_pose"] = "BODY_25"
params["fps_max"] = 60
params['write_video_fps']=-1
params['number_people_max']= 1
#params["disable_blending"] = True

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())


# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create objects to process pictures
datum = op.Datum()

# Create array to save all keypoint frame
KeypointFrame=np.array([])
image_count=0
start = time.time()


while vs.isOpened():


    #Get frame from video or webcam
    ret ,frame=vs.read()
    if not ret:
        break
    #Give inputData for openpoes to process
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    #Get openpose Output
    image = datum.cvOutputData
    cv2.imshow("Openpose", image)

    image_count+=1

    if vs.get(cv2.CAP_PROP_POS_FRAMES)==index+39:
        cv2.waitKey(0)
    if cv2.waitKey(1)  == ord('g'):
        print(vs.get(cv2.CAP_PROP_POS_FRAMES))
    if cv2.waitKey(1)  == ord('q'):
        break


end=time.time()
total_time=end-start
print('FPS:',image_count/total_time)


vs.release()
cv2.destroyAllWindows()






