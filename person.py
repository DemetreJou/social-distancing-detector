import numpy as np
from config import social_distancing_min_distance

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

    def check_social_distancing(self, other):
        social_distancing = self.get_distance_between_persons(other) > social_distancing_min_distance
        return social_distancing

    def set_social_distancing(self, social_distancing):
        self.social_distancing = social_distancing