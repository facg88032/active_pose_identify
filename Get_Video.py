import cv2

cap = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'MJPG')



cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# 建立 VideoWriter 物件，輸出影片至 output.avi
# FPS 值為 20.0，解析度為 640x360
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (1280,720))

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:

    # 寫入影格
    out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break

# 釋放所有資源
cap.release()
out.release()
cv2.destroyAllWindows()