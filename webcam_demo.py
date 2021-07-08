from main import process_frame
from waggle.data.vision import Camera
import cv2

cam = Camera()

for sample in cam.stream():
    cv2.imshow("capture", sample.data)
    cv2.waitKey(1)
    print(process_frame(sample.data))
