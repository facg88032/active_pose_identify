import cv2
from openpose import pyopenpose as op
import numpy as np
import time
import os

Video='process_basketball_Video/dribble/'+'d9.mp4'
vs=cv2.VideoCapture(Video)

width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["model_pose"] = "BODY_25"
params["fps_max"] = -1
params['write_video_fps']=-1
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

    #Check  openpose whether detect keypoints or not
    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:

        #Reshape keypoints data and save KeypointFrame
        keypoints=datum.poseKeypoints[0].reshape(1, 25,3)
        if KeypointFrame.size !=0:
            KeypointFrame = np.vstack((KeypointFrame, keypoints))
        else:
            KeypointFrame = keypoints

    #Get openpose Output
    image = datum.cvOutputData
    image_count +=1
    #Show the output
    cv2.imshow("Openpose", image)
    if cv2.waitKey(1)  == ord('q'):
        break

end=time.time()
total_time=end-start
print('FPS:',image_count/total_time)


vs.release()
cv2.destroyAllWindows()






