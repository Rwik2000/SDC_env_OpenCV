import numpy as np
import cv2
import time
from vehicle import Vehicle
from track import Track
import random

class env():
    def __init__(self, speed_X=1, dist_to_px = 20, render = 0):
        self.speed_X = speed_X
        self.dist_to_px = dist_to_px
    
    def gen_vehicle(self):
        v = Vehicle(speed_X=self.speed_X, dist_to_px=self.dist_to_px)
        return v
    
    def gen_track(self):
        m = Track(self.dist_to_px)
        m.dist_to_px = self.dist_to_px
        return m





