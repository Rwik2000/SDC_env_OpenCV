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
        # self.render = render
    
    def gen_vehicle(self):
        v = Vehicle(speed_X=self.speed_X, dist_to_px=self.dist_to_px)
        # v.render = self.render
        return v
    
    def gen_track(self):
        m = Track(self.dist_to_px)
        m.dist_to_px = self.dist_to_px
        return m

ENV = env(speed_X=300)
trk01 = ENV.gen_track()
Car01 = ENV.gen_vehicle()


for i in range(5):
    trk01_scr, spawn_loc = trk01.gen_track()
    done = 0
    outcome="DEAD"
    Car01.vel = 20*ENV.speed_X
    start = time.time()
    locus = trk01_scr.copy()
    while done == 0:
        input_scr = trk01_scr.copy()
        Car01.track = input_scr
        Car01.loc = spawn_loc
        spawn_loc,done=Car01.move(0,0)
        xyz = np.array(spawn_loc).astype(int)
        cv2.circle(locus, tuple(xyz), 3, (255,0,0), 2)
        cv2.imshow("locus",locus)
        cv2.waitKey(1)

        '''If render not required, comment the next line'''
        Car01.render()
        if spawn_loc[1]<0:
            outcome = "SUCCESS"
            break
    print(time.time()-start)
    cv2.circle(locus, tuple(xyz), 3, (255,0,0), 2)
    cv2.imshow("locus",locus)
    cv2.waitKey(0)
    print(outcome)
    print("-----------------------------")



