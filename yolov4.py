import numpy as np
import cv2
import time
import os
import itertools
from person import Person
from config import *

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
scale_h, scale_w = floor_depth/height, floor_width/width

def scale_box(box):
    box = list(box)
    box[0] = int(box[0] *scale_w)
    box[2] = int(box[2] *scale_w)
    box[1] = int(box[1] *scale_h)
    box[3] = int(box[3] *scale_h)
    return box

def scale_coords(coords):
    return (int(coords[0]*scale_w), int(coords[1]*scale_h)) 

###### Setup the video writer for the combined video detection + 2d floor location points ######
ground_video = cv2.VideoWriter(f'./{output_folder}/{file_name}_ground.avi', fourcc, fps, (floor_width*2, floor_depth))


frame_number = 0
while vc.isOpened():
    detected_people = []
    red_pairs = []
    
    (grabbed, frame) = vc.read()
    if not grabbed:
        print("not grabbed")
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # Resize frame
    frame = cv2.resize(frame, (floor_depth, floor_width))

    # Init a gray ground plane image 
    ground_plane = np.ones((floor_depth, floor_width, 3), np.uint8) * 20

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        # color = COLORS[int(classid) % len(COLORS)]
        detected_class = class_names[classid[0]]
        if detected_class == "person":
            person = Person(score, box)
            person.set_ground_coordinates(M)
            detected_people.append(person)
  
    # check for social distancing
    for (person, other) in itertools.combinations(detected_people, 2):
        social_distancing = person.check_social_distancing(other)
        if not social_distancing:
            person.set_social_distancing(False)
            other.set_social_distancing(False)
            red_pairs.append((person, other))

    # Draw the The red/green boxes and circles
    for person in detected_people:
        label = "%s : %f" % ("person", person.score)
        color = green if person.social_distancing else red

        # draw a rectangle around the person on the image frame
        box = scale_box(person.box)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 

        # draw a circle where the person is on the ground plane frame
        cv2.circle(ground_plane, (person.ground_x, person.ground_y+700), 15, color, -1)

    # Draw the red lines between people not social distancing
    for (person, other) in red_pairs:
        person_mid = scale_coords(person.middle)
        other_mid = scale_coords(other.middle)
        cv2.line(frame, person_mid, other_mid, red, 2) 

        person_ground_coords = (person.ground_x, person.ground_y+700)
        other_ground_coords = (other.ground_x, other.ground_y+700)
        cv2.line(ground_plane, person_ground_coords, other_ground_coords, red, 2) 

    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    (text_width, text_height) = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)[0]
    box_coords = ((0, 40), (0 + text_width + 5, 25 - text_height - 5)) # (bottom left corner, upper right corner) I THINK!!?!
    cv2.rectangle(frame, box_coords[0], box_coords[1], black, cv2.FILLED)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
    print(frame_number, fps_label)

    # Save the image frame
    # cv2.imwrite(f"./{output_folder}/{file_name}_{frame_number}.jpg", frame)
    
    # Write the frame to the videowriter
    combined_frames = np.hstack((frame, ground_plane))
    video_writer.write(frame)
    ground_video.write(combined_frames)

    # Show the frame with imshow, TODO: can change, doesn't work on wsl without extra software
    # cv2.imshow("detections", frame)

    frame_number += 1
    # if frame_number > 125:  # for quick testing
    #     break

video.release()
ground_video.release()