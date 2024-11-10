import ultralytics
import cv2
from PIL import Image
import numpy as np

gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")


ultralytics.checks()

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

#results = model(source = 0, show=True, conf=0.3, save=True)

#video_source = "personbottle.mp4"
#video_source = 0
video_source = gst_str

cap = cv2.VideoCapture(video_source, cv2.CAP_GSTREAMER)
if cap.isOpened():
    print("Video Opened")
else:
    print("Video Not Opened")
    print("Program Abort")
    exit()
#cv2.namedWindow("YOLOv8", cv2.WINDOW_GUI_EXPANDED)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #cv2.imshow("RAW", frame)
        
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8", annotated_frame)
        
    else:
        break
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


