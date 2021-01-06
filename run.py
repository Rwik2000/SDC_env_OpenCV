import time
import numpy as np
import cv2

from environment import env

ENV = env(speed_X=30)
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
        spawn_loc,done=Car01.move(0,0.06)
        xyz = np.array(spawn_loc).astype(int)
        '''If render not required, comment the next line'''
        Car01.render()
        if spawn_loc[1]<0:
            outcome = "SUCCESS"
            break
    print(time.time()-start)
    print(outcome)
    print("-----------------------------")