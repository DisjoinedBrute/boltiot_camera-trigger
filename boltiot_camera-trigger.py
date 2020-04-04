from imageai.Detection import VideoObjectDetection
import os
import cv2
from boltiot import Bolt
import time

api_key = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
device_id = "BOLT1234"
mybolt = Bolt(api_key, device_id)
state = False


def video_analysis(frame_number, output_array, output_count):
    global state
    if bool(output_array) is True and state == False:

        print("Room Occupied")
        time.sleep(10)
        response = mybolt.digitalWrite('2', 'LOW')
        state = True
    elif bool(output_array) is False and state == True:

        print("Room not occupied")
        time.sleep(10)
        response = mybolt.digitalWrite('2', 'HIGH')
        state = False


execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, camera_input=camera,
                                                   save_detected_video=False, frames_per_second=20, log_progress=False,
                                                   minimum_percentage_probability=40, per_frame_function=video_analysis)
