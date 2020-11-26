import cv2
from openpose import pyopenpose as op
import keras
import numpy as np
import random
import time
import tensorflow as tf
import pandas as pd
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

mpose = keras.models.load_model('weights-improvement-38-1.00.hdf5')

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())

Video='process_basketball_Video/dribble/'+'d10.mp4'
vs=cv2.VideoCapture(Video)

width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Starting OpenPoseasdasdas
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()

fps_time = 0


# Create array to save all keypoint frame
KeypointFrame=np.array([])
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

    if len(KeypointFrame)==50:
        Append_list = random.sample(range(50), 30)
        Append_list.sort()
        process_data = []
        for i in Append_list:
            process_data.append(KeypointFrame[i])
        print('123')
        process_data = np.asarray(process_data).reshape(1,30,75)
        # process_data = pd.DataFrame(process_data)
        #
        #
        # def normalize(train):
        #
        #     train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        #     return train_norm
        #
        #
        # process_data = normalize(process_data)
        # process_data = process_data.values
        # blocks = int(len(process_data) / 30)
        # process_data = np.array(np.split(process_data, blocks))
        output = mpose.predict_classes(process_data)
        print(output)
        KeypointFrame=np.array([])
    #Gt openpose Output
    image = datum.cvOutputData
    #Show the output
    #cv2.imshow("Openpose", image)
    if cv2.waitKey(1)  == ord('q'):
        break


# clean up after yourself
vs.release()
cv2.destroyAllWindows()


