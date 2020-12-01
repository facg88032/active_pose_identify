import cv2

vs=cv2.VideoCapture('44.mp4')


while True:

    _,frame=vs.read()

    cv2.imshow('sss',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


vs.release()

cv2.destroyAllWindows()