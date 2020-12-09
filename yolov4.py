import numpy as np
import cv2
import time
import os
from person import Person

###### Model Detection Parameters ######
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

###### Drawing shapes variables and colours ######
green = (0, 255, 0)
red = (0, 0, 255)
social_distancing_min_distance = 200

###### Get the list of trained classes our model config/weights file ######
with open("./darknet/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


###### Choose the video we want to run on ######
file_name, fps = "EnterExitCrossingPaths2front_10fps", 10
# file_name, fps = "EnterExitCrossingPaths2front_shortened", 25
extention = "mp4"

###### Setup the relative input/output folders ######
relative_path = f"./Videos/{file_name}.{extention}"
output_folder = "Output"
os.makedirs(output_folder, exist_ok=True)

###### Setup the video capture and video writer objects ######
vc = cv2.VideoCapture(relative_path)
(grabbed, frame) = vc.read() # Steal the first frame to get the size hahaha 
height, width = frame.shape[0], frame.shape[1]
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video_writer = cv2.VideoWriter(f'./{output_folder}/{file_name}_detection.avi', fourcc, fps, (width,height))

###### Setup the detection model with preferable gpu usage ######
net = cv2.dnn.readNet("darknet/yolov4.weights", "darknet/cfg/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


###### Fit the homography matrix mapping for the sample mall video ######
src_pts = np.array([[60, 153], [359, 153], [50, 201], [367, 200]])
dst_pts = np.array([[0, 0], [0, 975], [382, 98], [382, 878]])
floor_depth, floor_width =  1100, 1100
M, mask = cv2.findHomography(src_pts, dst_pts, 0)

###### Setup the video writer for the combined video detection + 2d floor location points ######
ground_video = cv2.VideoWriter(f'./{output_folder}/{file_name}_ground.avi', fourcc, fps, (floor_width*2, floor_depth))


frame_number = 0
while vc.isOpened():
    detected_people = []
    (grabbed, frame) = vc.read()
    if not grabbed:
        print("not grabbed")
        exit()

    # Init a black ground plane image 
    ground_plane = np.ones((floor_depth, floor_width, 3), np.uint8) * 20

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        # color = COLORS[int(classid) % len(COLORS)]
        detected_class = class_names[classid[0]]
        if detected_class == "person":
            person = Person(score, box)
            person.set_ground_coordinates(M)
            detected_people.append(person)
  
    # check for social distancing
    for person in detected_people:
        for other in detected_people:
            if person != other:
                person.set_social_distancing(other)

    # Draw the shapes on the frame
    for person in detected_people:
        label = "%s : %f" % ("person", person.score)
        color = green if person.social_distancing else red

        # draw a rectangle around the person on the image frame
        cv2.rectangle(frame, person.box, color, 2)
        cv2.putText(frame, label, person.middle, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2) # (box[0], box[1] - 10) (old text location)

        # draw a circle where the person is on the ground plane frame
        cv2.circle(ground_plane, (person.ground_x, person.ground_y+700), 15, color, -1)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    print(frame_number, fps_label)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

    # Save the image frame
    # cv2.imwrite(f"./{output_folder}/{file_name}_{frame_number}.jpg", frame)
    
    # Write the frame to the videowriter
    combined_frames = np.hstack((cv2.resize(frame, (floor_depth, floor_width)), ground_plane))
    video_writer.write(frame)
    ground_video.write(combined_frames)

    # Show the frame with imshow, TODO: can change, doesn't work on wsl without extra software
    # cv2.imshow("detections", frame)

    frame_number += 1
    # if frame_number > 125:  # for quick testing
    #     break

video.release()
ground_video.release()