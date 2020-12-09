import os

###### Model Detection Parameters ######
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

###### Drawing shapes variables and colours ######
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)
social_distancing_min_distance = 200

###### Get the list of trained classes our model config/weights file ######
with open("./darknet/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]