import cv2
from openpose import pyopenpose as op
import keras
import numpy as np
import random
import time
from sklearn import preprocessing
import tensorflow as tf
from Label_method.utils import Utils
import pandas as pdg
import configparser
import argparse
import joblib as jb

def load_Op_model(config):
    params = dict()
    params["model_folder"] = config['model_folder']
    params["model_pose"] = config['model_pose']
    params["fps_max"] = config.getint('fps_max')
    params['write_video_fps'] = config.getint('write_video_fps')
    params['number_people_max'] = config.getint('number_people_max')
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()
    return opWrapper,datum

def load_HAR_model(config):
    model=keras.models.load_model(config['model_path'])
    return model

def load_video(config):
    vs=cv2.VideoCapture(config['video_path'])
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint('width'))
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint('height'))
    vs.set(cv2.CAP_PROP_POS_FRAMES, config.getint('index'))
    return  vs

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def main():

    cfg=load_config()

    # poseModel = op.PoseModel.BODY_25
    # original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
    # keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())

    #Video='V1.mp4'
    #Video='process_basketball_Video/shoot/s104.mp4'
    vs=load_video(cfg['video'])
    HAR=load_HAR_model(cfg['har'])

    # Starting OpenPose
    wrapper,datum=load_Op_model(cfg['openpose'])
    max_frame=40

    util=Utils()

    # Create array to save all keypoint frame
    KeypointFrame=np.asarray([])
    start = time.time()

    dribble=[]
    shoot=[]

    d=0
    s=0
    # scaler = jb.load('training/model/model_ND1/std_scaleND.bin')
    scaler = jb.load('training/model/model_Final/std_scaleND.bin')

    while vs.isOpened():
        No_img=int(vs.get(cv2.CAP_PROP_POS_FRAMES))
        #Get frame from video or webcam
        ret ,frame=vs.read()
        if not ret:
            break

        #Give inputData for openpoes to process
        datum.cvInputData = frame
        wrapper.emplaceAndPop([datum])
        # Gt openpose Output
        image = datum.cvOutputData

        #Check  openpose whether detect keypoints or not
        if datum.poseKeypoints.any() and datum.poseKeypoints.ndim == 3:

            #Reshape keypoints data and save KeypointFrame
            keypoints=datum.poseKeypoints[0].reshape(1, 25,3)
            keypoints[:, :, 0] = keypoints[:, :, 0] / 640
            keypoints[:, :, 1] = keypoints[:, :, 1] / 480
            KeypointFrame=util.combine(KeypointFrame,keypoints)


        if len(KeypointFrame)==max_frame:
            SampleIndexs=util.StratifiedRandomSample(max_frame,30)
            sample_data = util.DataProcess(KeypointFrame, SampleIndexs)

            #標準化
            sample_data = np.asarray(sample_data).reshape(30,25*3)
            sample_data = scaler.transform(sample_data)
            sample_data = sample_data.reshape(1,30,75)

            #正規劃
            # sample_data = np.asarray(sample_data).reshape(30,25*3)
            # sample_data= preprocessing.normalize(sample_data)
            # blocks = int(len(sample_data) / 30)
            # sample_data = np.array(np.split(sample_data, blocks))

            result=HAR.predict(sample_data)
            print(result)
            threshold=0.9

            if result[0][0] > threshold:
                d+=1
                if d>=0:
                    cv2.putText(image,
                                "dribble",
                                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 25, 255), 5)
                    d=0
                start_img=No_img-39
                dribble.append(start_img)
            elif result[0][1] > threshold:
                s+=1
                if s>=0:
                    cv2.putText(image,
                                "shoot",
                                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (85, 255, 255), 5)
                    s=0
                start_img = No_img - 39
                shoot.append(start_img)
            else:
                d=0
                s=0

            KeypointFrame=np.delete(KeypointFrame,np.s_[:1],0)


        #Show the output
        cv2.imshow("Openpose", image)

        if cv2.waitKey(1)  == ord('q'):
            break



    # clean up after yourself
    vs.release()
    cv2.destroyAllWindows()

    with open('dribble_log'+'.txt', "w") as fs:
        for i in dribble:
            fs.write(str(i) + "\n")
    # with open('shoot_log25'+'.txt', "w") as fs:
    #     for i in shoot:
    #         fs.write(str(i) + "\n")


if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=tf_config)



    main()