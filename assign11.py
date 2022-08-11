import numpy as np
import cv2
import imutils
import time

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0,255,size = (len(CLASSES) , 3))

print("loading..........")
net = cv2.dnn.readNetFromCaffe(prototxt , model)
print("model loaded")
print("Starting camera feed..................")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _ , frame = vs.read()
    frame = imutils.resize(frame , width = 500)

    (h,w) = frame.shape[:2]
    imResizeBlob = cv2.resize(frame , (300,300))
    blob = cv2.dnn.blobFromImage(imResizeBlob , 0.007843 , (300,300) , 127.5)

    net.setInput(blob)
    detection = net.forward()
    detShape = detection.shape[2]
    
    for i in np.arange(0,detShape):
        confidence = detection[0,0,i,2]
        if confidence > confThresh:
            idx = int(detection[0,0,i,1])
            if idx == 5:
                print("I need Water")
            box = detection[0,0,i,3:7] * np.array([w, h, w, h])
            (sx,sy,ex,ey) = box.astype("int")

            label = "{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(sx,sy),(ex,ey),COLORS[idx],2)
            if sy-15 > 15:
                y = sy-15
            else:
                y = sy+15
            cv2.putText(frame,label,(sx,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

vs.release()
cv2.destroyAllWindows()
