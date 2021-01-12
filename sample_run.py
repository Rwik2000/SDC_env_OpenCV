import time
import numpy as np
import cv2
import random
from environment import env

ENV = env(speed_X=70)
trk01 = ENV.gen_track()
ENV.num_vehicles = 10
initial_vel = 20
Cars = ENV.gen_vehicles()
ENV.print_info = 1

for i in range(20):
    trk01_scr, spawn_loc = trk01.gen_track()
    outcome="DEAD"
    for i in range(ENV.num_vehicles):
        ENV.vehicles[i].vel = initial_vel*ENV.speed_X
    # print(ENV.vehicles[0].vel)
    # Car01.vel = 10*ENV.speed_X
    # Car02.vel = 10*ENV.speed_X
    # Car03.vel = 10*ENV.speed_X
    start = time.time()
    locus = trk01_scr.copy()
    tot_reward = 0
    cflag = 0
    while 1:
        input_scr = trk01_scr.copy()
        dones = np.zeros(ENV.num_vehicles)
        for i in range(ENV.num_vehicles):
            # temp_car = ENV.vehicles[i]
            if dones[i] ==0:
                ENV.vehicles[i].track = input_scr
                if cflag == 0:
                    ENV.vehicles[i].loc = spawn_loc.copy()

                '''Add your Code here'''
                throttle = 0.3
                steer = 0.5
                steer_val = random.choice([-1,1])*steer

                _ ,dones[i], reward = ENV.vehicles[i].move(1,steer_val)
                # print(ENV.vehicles[i].vis_pts)

        '''If render not required, comment the next line'''
        ENV.render()
        # print()
        cflag+=1
        # print(dones)
        # print("----------------------")
        if 0 not in dones:
            if 1 in dones:
                print("SUCCESS")
            else:
                print("FAILURE")
            print(time.time() - start)
            break

