import cv2
import time
import os

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

green = (0, 255, 0)
red = (0, 0, 255)

with open("./darknet/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# vc = cv2.VideoCapture("http://192.168.2.60:4747/video")  # TODO: for droid cam

file_name = "starbucks"  # to help with saving later
extention = "mp4"
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

    def __eq__(self, other):
        for a, b in zip(self.box, other.box):
            if a != b:
                return False
        return True

    @property
    def middle(self):
        return (self.box[0] + self.box[2]//2, self.box[1] + self.box[3]//2)

    @property
    def width(self):
        return self.box[1]

    @property
    def height(self):
        return self.box[0]

    def set_social_distancing(self, other):
        # TODO: implement
        pass


frame_number = 0
while vc.isOpened():
    detected_people = []
    (grabbed, frame) = vc.read()
    if not grabbed:
        print("not grabbed")
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        # color = COLORS[int(classid) % len(COLORS)]
        detected_class = class_names[classid[0]]
        if detected_class == "person":
            detected_people.append(Person(score, box))

    # check for social distancing
    for person in detected_people:
        for other in detected_people:
            if person != other:
                person.set_social_distancing(other)

    for person in detected_people:
        label = "%s : %f" % ("person", person.score)
        color = green if person.social_distancing else red
        cv2.rectangle(frame, person.box, color, 2)
        # (box[0], box[1] - 10)
        cv2.putText(frame, label, person.middle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    print(fps_label)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(f"./{output_folder}/{file_name}_{frame_number}.jpg", frame)
    # cv2.imshow("detections", frame)   # TODO: can change, doesn't work on wsl without extra software
    frame_number += 1
    if frame_number > 60:  # for quick testing
        break
