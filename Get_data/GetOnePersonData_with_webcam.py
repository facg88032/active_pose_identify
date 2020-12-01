import cv2
from openpose import pyopenpose as op
import numpy as np

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

# Create array to save all keypoint frame
KeypointFrame=np.array([])

while (vs.isOpened()):
    #Get frame from video or webcam
    ret ,frame=vs.read()
    #Give inputData for openpoes to process
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    #Check  openpose whether detect keypoints or not
    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:

        #Reshape keypoints data and save KeypointFrame
        keypoints=datum.poseKeypoints[0].reshape(1, 75)
        if KeypointFrame.size !=0:
            KeypointFrame = np.vstack((KeypointFrame, keypoints))
        else:
            KeypointFrame = keypoints

    #Get openpose Output
    image = datum.cvOutputData
    #Show the output
    cv2.imshow("Openpose", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print(KeypointFrame.shape)

#Save data  as Numpy type
np.save('data.npy',KeypointFrame)
vs.release()
cv2.destroyAllWindows()

