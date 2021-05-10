import cv2
from openpose import pyopenpose as op
import numpy as np
import time
import os
import argparse

def main(video_path,save_path):
    for video in os.listdir(video_path):
        # if video.endswith('.avi'):
        if video.endswith('.mp4'):
            print('Processing'+video+'...........')
            vs=cv2.VideoCapture(video_path+'/'+video)

            width=1280
            height=720
            vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = "../../../../models/"
            params["model_pose"] = "BODY_25"
            params["fps_max"] = -1
            params['write_video_fps']=-1
            params['number_people_max'] = 1
            poseModel = op.PoseModel.BODY_25
            original_keypoints_index = op.getPoseBodyPartMapping(poseModel)
            # keypoints_index = dict((bp, num) for num, bp in original_keypoints_index.items())


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
                # image = datum.cvOutputData
                image_count +=1
                #Show the output
                #cv2.imshow("Openpose", image)
                if cv2.waitKey(1)  == ord('q'):
                    break

            end=time.time()
            total_time=end-start
            print('FPS:',image_count/total_time)

            #Save data  as Numpy type
            # np.save(save_path+'/'+video.replace(".mp4",".npy"),KeypointFrame)
            np.save(save_path + '/' + video.replace(".mp4", ".npy"), KeypointFrame)
            print('Successful get '+video+' KeyPoints')

            vs.release()
            cv2.destroyAllWindows()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../process_basketball_Video", help="input the video path")
    parser.add_argument("--save_path", default="../process_basketball_Video", help="input the video path")
    args = parser.parse_known_args()
    main(args[0].video_path,args[0].save_path)





