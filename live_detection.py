from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import itertools
import sys
from heatmap import Heatmap
from webcam import VideoScreenshot
from person import Person
from config import *

# Check for inputted video source, otherwise use default
if len(sys.argv) > 1:
    src = sys.argv[1]
    exit
else:
    src = "http://192.168.2.13:4747/video"

# Instantiate the live video reader to read lastest frames
vc = VideoScreenshot(src)

###### User homography setup stage ######
grabbed = False
while not grabbed:
    (grabbed, frame) = vc.read()
height, width = frame.shape[0], frame.shape[1]
print(f'height={height}, width={width}')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
larger_frame = np.pad(frame[:,:,::-1], ((height//2,), (width//2,), (0,)))
ax.imshow(larger_frame, origin='upper', extent=[-width//2, width+width//2, height+height//2, -height//2])

cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
plt.draw()

text = ('Select top-left, top-right, bottom-left and bottom-right corners with mouse')
plt.title(text, fontsize=16)
pts = np.asarray(plt.ginput(4))
plt.close()
print(pts)


################################################################
################################################################
################################################################

###### Setup the detection model with preferable gpu usage ######
net = cv2.dnn.readNet("darknet/yolov4.weights", "darknet/cfg/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

###### Fit the homography matrix mapping for the sample mall video ######
src_pts = pts
dst_pts = np.array([[0, 0], [0, width], [height, 0], [height, width]])
floor_depth, floor_width =  height, width
M, mask = cv2.findHomography(src_pts, dst_pts, 0)
M_inv, mask_inv = cv2.findHomography(np.array([[0, 0], [width, 0], [0, height], [width, height]]), src_pts, 0)
scale_h, scale_w = floor_depth/height, floor_width/width

# historical_img = np.zeros((floor_depth, floor_width, 1))
# def add_to_heatmap(img):
#     if img.ndim == 3:
#         img = np.mean(img, 2)

#     img[img==grey_val] = 0 # Remove the grey background
#     img[img>0] = 1
#     img = img[::-1,].reshape(img.shape + (1,))

#     global historical_img
#     historical_img = np.concatenate((historical_img, img), axis=2)

# fig, ax = plt.subplots()
# def draw_heatmap(frame_number):
#     heatmap = gaussian_filter(np.sum(historical_img, 2), 1)
#     ax.pcolormesh(heatmap, cmap='seismic', vmin=-historical_img.max(), vmax=historical_img.max(), alpha=0.5)
#     fig.savefig(f"./{output_folder}/colormap_frame_{frame_number}.jpg")
#     ax.clear()

detection_heatmap = Heatmap(floor_depth, floor_width)


def scale_box(box):
    box = list(box)
    box[0] = int(box[0] *scale_w)
    box[2] = int(box[2] *scale_w)
    box[1] = int(box[1] *scale_h)
    box[3] = int(box[3] *scale_h)
    return box

def scale_coords(coords):
    return (int(coords[0]*scale_w), int(coords[1]*scale_h)) 


frame_number = 0
while cv2.waitKey(1) < 1:
    detected_people = []
    red_pairs = []
    
    (grabbed, frame) = vc.read()
    if not grabbed:
        print("not grabbed")
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # # Resize frame
    # frame = cv2.resize(frame, (floor_depth, floor_width))

    # Init a ground plane image 
    ground_plane = np.zeros((floor_depth, floor_width, 3), np.uint8)

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
        if color == green:
            continue

        # draw a rectangle around the person on the image frame
        box = scale_box(person.box)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 

        # draw a circle where the person is on the ground plane frame
        ground_pos = (person.ground_x, person.ground_y)
        cv2.circle(ground_plane, (person.ground_x, person.ground_y), 15, color, -1)

    # Draw the red lines between people not social distancing
    for (person, other) in red_pairs:
        person_mid = scale_coords(person.middle)
        other_mid = scale_coords(other.middle)
        cv2.line(frame, person_mid, other_mid, red, 2) 

        person_ground_coords = (person.ground_x, person.ground_y)
        other_ground_coords = (other.ground_x, other.ground_y)
        cv2.line(ground_plane, person_ground_coords, other_ground_coords, red, 2) 

    # Add the current frame to the heatmap, then save or show heatmap
    detection_heatmap.add_frame_to_heatmap(ground_plane.copy())
    # detection_heatmap.save_heatmap_plot(f"./Output/colormap_frame_{frame_number}.jpg", frame, M_inv)
    detection_heatmap.show_heatmap_plot(frame, M_inv)

    # Draw the The red/green boxes and circles
    for person in detected_people:
        label = "%s : %f" % ("person", person.score)
        color = green if person.social_distancing else red
        if color == red:
            continue

        # draw a rectangle around the person on the image frame
        box = scale_box(person.box)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 

        # draw a circle where the person is on the ground plane frame
        ground_pos = (person.ground_x, person.ground_y)
        cv2.circle(ground_plane, (person.ground_x, person.ground_y), 15, color, -1)
    
    # Add the grey background
    ground_plane[ground_plane == 0] = grey_val

    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    (text_width, text_height) = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)[0]
    box_coords = ((0, 40), (0 + text_width + 5, 25 - text_height - 5)) # (bottom left corner, upper right corner) I THINK!!?!
    cv2.rectangle(frame, box_coords[0], box_coords[1], black, cv2.FILLED)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2)
    # print(frame_number, fps_label)

    # Save the image frame
    # cv2.imwrite(f"./{output_folder}/{file_name}_{frame_number}.jpg", frame)
    
    # Combine the frames side by side
    combined_frames = np.hstack((frame, ground_plane))

    # Show the frame with imshow, TODO: can change, doesn't work on wsl without extra software
    cv2.imshow("detections", combined_frames)

    frame_number += 1
    if frame_number % 60 == 0:
        detection_heatmap.save_heatmap_plot(f"./Output/colormap_frame_{frame_number}.jpg", frame, M_inv)
