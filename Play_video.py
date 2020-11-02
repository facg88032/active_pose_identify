import cv2

vs=cv2.VideoCapture('output.avi')


while True:

    _,frame=vs.read()

    cv2.imshow('sss',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


vs.release()

cv2.destroyAllWindows()