import numpy as np
import cv2
import time
import os

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

green = (0, 255, 0)
red = (0, 0, 255)
social_distancing_min_distance = 200

with open("./darknet/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# vc = cv2.VideoCapture("http://192.168.2.60:4747/video")  # TODO: for droid cam

file_name = "EnterExitCrossingPaths2front_10fps"  # to help with saving later
extention = "mp4"
fps=10
relative_path = f"./Videos/{file_name}.{extention}"
output_folder = "Output"

os.makedirs(output_folder, exist_ok=True)

vc = cv2.VideoCapture(relative_path)

net = cv2.dnn.readNet("darknet/yolov4.weights", "darknet/cfg/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


class Person:
    score: int
    box: list
    social_distancing: bool

    def __init__(self, score, box):
        self.score = score
        self.box = box
        self.social_distancing = True
        self.ground_x = None
        self.ground_y = None

    def __eq__(self, other):
        for a, b in zip(self.box, other.box):
            if a != b:
                return False
        return True
    
    def set_ground_coordinates(self, M):
        pt = np.append(self.feet, [1])
        mapped_pt = np.dot(M, pt)
        self.ground_y = int(mapped_pt[0] / mapped_pt[2])
        self.ground_x = int(mapped_pt[1] / mapped_pt[2])

    @property
    def middle(self):
        # middle of the box 
        return (self.box[0] + self.box[2]//2, self.box[1] + self.box[3]//2)

    @property
    def feet(self):
        # bottom middle of the box (ie where the persons feet are in (col, row)/(x,y) )
        return (self.box[0] + self.box[2]//2, self.box[1] + self.box[3])

    @property
    def ground_coordinates(self):
        # bottom middle of the box (ie where the persons feet are in (col, row)/(x,y) )
        return (self.ground_x, self.ground_y)

    @property
    def width(self):
        return self.box[1]

    @property
    def height(self):
        return self.box[0]

    def get_distance_between_persons(self, other):
        dist = np.sqrt((self.ground_x - other.ground_x)**2 + (self.ground_y - other.ground_y)**2)
        return dist

    def set_social_distancing(self, other):
        if self.get_distance_between_persons(other) < social_distancing_min_distance:
            self.social_distancing = False

        

# Setup the video writer
(grabbed, frame) = vc.read() #Steal the first frame to get the size hahaha
height, width = frame.shape[0], frame.shape[1]
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter(f'./{output_folder}/{file_name}_detection.avi', fourcc, fps, (width,height))

# Fit the homography matrix mapping for the sample mall video
src_pts = np.array([[60, 153], [359, 153], [50, 201], [367, 200]])
dst_pts = np.array([[0, 0], [0, 975], [382, 98], [382, 878]])
floor_depth, floor_width =  1000, 1000
M, mask = cv2.findHomography(src_pts, dst_pts, 0)

# TEST THE HOMOGRAPHY
new_img = np.zeros((floor_depth, floor_width, 3))

for r in range(height):
    for c in range(width):
        x = c
        y = r
        pt = np.array([x,y,1])
        mapped_pt = np.dot(M, pt)
        mapped_r = int(mapped_pt[0] / mapped_pt[2])
        mapped_c = int(mapped_pt[1] / mapped_pt[2])

        if 0 <= mapped_c < floor_width and 0 <= mapped_r < floor_depth:
            new_img[mapped_r, mapped_c] = frame[r,c]

cv2.imwrite(f'{output_folder}/test_homography.jpg', new_img[:,:,::-1])

ground_video = cv2.VideoWriter(f'./{output_folder}/{file_name}_ground.avi', fourcc, fps, (floor_width, floor_depth))

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
        cv2.circle(ground_plane, (person.ground_x, person.ground_y+600), 15, color, -1)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    print(frame_number, fps_label)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

    # Save the image frame
    # cv2.imwrite(f"./{output_folder}/{file_name}_{frame_number}.jpg", frame)

    # Write the frame to the videowriter
    video.write(frame)
    ground_video.write(ground_plane)

    # Show the frame with imshow, TODO: can change, doesn't work on wsl without extra software
    # cv2.imshow("detections", frame)

    frame_number += 1
    # if frame_number > 125:  # for quick testing
    #     break

video.release()
ground_video.release()