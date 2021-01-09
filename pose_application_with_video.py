import cv2
from openpose import pyopenpose as op
import keras
import numpy as np
import random
import time
import tensorflow as tf
import pandas as pd
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.InteractiveSession(config=config)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["model_pose"] = "BODY_25"
params["fps_max"] = -1
params['write_video_fps']=-1
#params["disable_blending"] = True
mpose = keras.models.load_model('training/weights-improvement-97-0.99.hdf5')

poseModel = op.PoseModel.BODY_25
original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())
Video='basketball_shot480p/sv22.mp4'

#Video='process_basketball_Video/dribble/'+'44.mp4'
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
max_frame=40

# Create array to save all keypoint frame
KeypointFrame=np.asarray([])
start = time.time()

dribble=[]
shoot=[]
combine_data=np.asarray([])
while vs.isOpened():
    No_img=int(vs.get(cv2.CAP_PROP_POS_FRAMES))
    #Get frame from video or webcam
    ret ,frame=vs.read()
    if not ret:
        break

    #Give inputData for openpoes to process
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    # Gt openpose Output
    image = datum.cvOutputData

    #Check  openpose whether detect keypoints or not
    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:

        #Reshape keypoints data and save KeypointFrame
        keypoints=datum.poseKeypoints[0].reshape(1, 25,3)
        keypoints[:, :, 0] = keypoints[:, :, 0] / 480
        keypoints[:, :, 1] = keypoints[:, :, 1] / 640
        if KeypointFrame.size !=0:
            KeypointFrame = np.vstack((KeypointFrame, keypoints))
        else:
            KeypointFrame = keypoints


    if len(KeypointFrame)==max_frame:



        # Append_list = random.sample(range(max_frame), 30)
        # Append_list.sort()
        # temp = []
        # for i in Append_list:
        #     temp.append(KeypointFrame[i])
        # resize x_data
        # resize x_data

        process_data = np.asarray(KeypointFrame).reshape(40,75)
        process_data = pd.DataFrame(process_data)


        def normalize(train):

            train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            return train_norm


        process_data = normalize(process_data)
        process_data = process_data.values
        blocks = int(len(process_data) / 40)
        process_data = np.array(np.split(process_data, blocks))
        output = mpose.predict_classes(process_data)
        print(output)
        # if output == 0:
        #     cv2.putText(image,
        #                 "dribble",
        #                 (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #                 (255, 25, 255), 5)
        #     start_img=No_img-39
        #     dribble.append(start_img)
        # elif output == 1:
        #     cv2.putText(image,
        #                 "shoot",
        #                 (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
        #                 (85, 255, 255), 5)
        #     start_img = No_img - 39
        #     shoot.append(start_img)
        #
        #

        KeypointFrame=np.delete(KeypointFrame,np.s_[:1],0)
        print(11)
        #time.sleep(1)

    #Show the output
    cv2.imshow("Openpose", image)

    if cv2.waitKey(1)  == ord('q'):
        break


# clean up after yourself
vs.release()
cv2.destroyAllWindows()
#
# with open('dribble_log'+'.txt', "w") as fs:
#     for i in dribble:
#         fs.write(str(i) + "\n")
# with open('shoot_log'+'.txt', "w") as fs:
#     for i in shoot:
#         fs.write(str(i) + "\n")
