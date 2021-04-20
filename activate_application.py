import cv2
from openpose import pyopenpose as op
import numpy as np
import Make_data_from_Keypointframe as Mk
import pandas as pd
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

def normalize(train):
  df=pd.DataFrame(data=train)
  train_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm


mpose = keras.models.load_model('weights-improvement-139-0.75.hdf5')


vs=cv2.VideoCapture(0)

width=1280
height=720
vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["model_pose"] = "COCO"
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
threshold=32
while (vs.isOpened()):
    #Get frame from video or webcam
    ret ,frame=vs.read()
    #Give inputData for openpoes to process
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    #Check  openpose whether detect keypoints or not
    if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3 and len(datum.poseKeypoints) <2:

        #Reshape keypoints data and save KeypointFrame
        #keypoints=datum.poseKeypoints[0].reshape(1, 75)

        keypoints=datum.poseKeypoints
        keypoints=datum.poseKeypoints[:, :, :2]
        keypoints=keypoints.reshape(1, 36)
        if KeypointFrame.size !=0:
            KeypointFrame = np.vstack((KeypointFrame, keypoints))
        else:
            KeypointFrame = keypoints


        if KeypointFrame.shape[0]==threshold:
            KeypointFrame = normalize(KeypointFrame)
            KeypointFrame = KeypointFrame.values.reshape(1,threshold,36)
            print(KeypointFrame.shape)

            output = mpose.predict_classes(KeypointFrame)
            print(output)
            KeypointFrame = np.array([])


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






