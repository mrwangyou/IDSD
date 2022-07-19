import math
import os
import random
import sys
import time

import jsbsim
import numpy as np

sys.path.append(str(jsbsim.get_default_root_dir()) + '/oFCM/')


class JsbsimEnv():

    def __init__(self,
                 fdm1_aircraft = 'f16',   fdm2_aircraft = 'f16',

                 fdm1_ic_v     = 500,     fdm2_ic_v     = 500,
                 fdm1_ic_lat   = 0,       fdm2_ic_lat   = 0.001,
                 fdm1_ic_long  = 0,       fdm2_ic_long  = 0.00005,
                 fdm1_ic_h     = 30005.5, fdm2_ic_h     = 30005.5,
                 fdm1_ic_psi   = 0,       fdm2_ic_psi   = 0,
                 fdm1_ic_theta = 0,       fdm2_ic_theta = 0,
                 fdm1_ic_phi   = 90,      fdm2_ic_phi   = 90,

                 fdm1_hp       = 1,       fdm2_hp      = 1,
                 fdm1_fgfs     = True,    fdm2_fgfs    = True,
                #  sim_speed     = 0,  # 0 for no delay
                 flight_mode   = 1,  # 0 for flight test
                 ) -> None:

        # FDM Initialization
        self.fdm1 = jsbsim.FGFDMExec(None)
        self.fdm2 = jsbsim.FGFDMExec(None)
        self.fdm = [self.fdm1, self.fdm2]
        
        self.fdm1.load_model(fdm1_aircraft)  # Aircraft
        self.fdm2.load_model(fdm2_aircraft)  
        
        if fdm1_fgfs is True:
            self.fdm1.set_output_directive('./data_output/flightgear.xml')  # Visualization fgfs Port 5550
        if fdm2_fgfs is True:
            self.fdm2.set_output_directive('./data_output/flightgear2.xml')  # Port 5551
        
        # Velocity Initialization
        self.fdm1['ic/vc-kts'] = fdm1_ic_v  # Calibrated Velocity (knots) https://skybrary.aero/articles/calibrated-airspeed-cas#:~:text=Definition,port%20caused%20by%20airflow%20disruption).
        self.fdm2['ic/vc-kts'] = fdm2_ic_v  # 1 knots = 1.852 km/h = 0.514 m/s

        # Position Initialization
        self.fdm1["ic/lat-gc-deg"] = fdm1_ic_lat  # Latitude (degree)
        self.fdm1["ic/long-gc-deg"] = fdm1_ic_long  # Longitude (degree)
        self.fdm1["ic/h-sl-ft"] = fdm1_ic_h  # Height above sea level (feet)
        self.fdm1["ic/psi-true-deg"] = fdm1_ic_psi  # 偏航角，绕Z轴，按照列出顺序进行欧拉角计算
        self.fdm1["ic/theta-deg"] = fdm1_ic_theta  # 俯仰角，绕Y轴
        self.fdm1["ic/phi-deg"] = fdm1_ic_phi  # 翻滚角，绕X轴

        self.fdm2["ic/lat-gc-deg"] = fdm2_ic_lat  # Recommend ~0.01
        self.fdm2["ic/long-gc-deg"] = fdm2_ic_long  # Recommend ~0.00005
        self.fdm2["ic/h-sl-ft"] = fdm2_ic_h
        self.fdm2["ic/psi-true-deg"] = fdm2_ic_psi
        self.fdm2["ic/theta-deg"] = fdm2_ic_theta
        self.fdm2["ic/phi-deg"] = fdm2_ic_phi

        ##########################
        ## Model Initialization ##
        self.fdm1.run_ic()      ##
        self.fdm2.run_ic()      ##
        ##########################

        # Engine Turning on
        self.fdm1["propulsion/starter_cmd"] = 1
        self.fdm2["propulsion/starter_cmd"] = 1

        # Refueling
        if flight_mode == 0:
            self.fdm1["propulsion/refuel"] = 1
            self.fdm2["propulsion/refuel"] = 1

        # First but not Initial
        self.fdm1.run()
        self.fdm1["propulsion/active_engine"] = True
        self.fdm1["propulsion/set-running"] = 0

        self.fdm2.run()
        self.fdm2["propulsion/active_engine"] = True
        self.fdm2["propulsion/set-running"] = 0

        self.nof = 1  # Number of frames

        # HP setting
        self.fdm1_hp = fdm1_hp
        self.fdm2_hp = fdm2_hp

    def getProperty(
        self,
        prop,
        id=0,
    ) -> list:
        if prop == 'position':
            prop = [
                "position/lat-gc-deg",  # Latitude 经度
                "position/long-gc-deg",  # Longitude 纬度
                "position/h-sl-ft",  # Altitude above sea level 海拔
            ]
        elif prop == 'positionEci':  # Earth-centered inertial
            prop = [
                "position/eci-x-ft",  # 指向经纬度为0的点
                "position/eci-y-ft",  # 左手系决定
                "position/eci-z-ft",  # 指向北极点
            ]
        elif prop == 'attitudeRad':
            prop = [
                "attitude/psi-rad",  # Yaw 偏航角
                "attitude/theta-rad",  # Pitch 俯仰角
                "attitude/phi-rad",  # Roll 翻滚角
            ]
        elif prop == 'pose':
            prop = [
                "position/lat-gc-deg",
                "position/long-gc-deg",
                "position/h-sl-ft",
                "attitude/psi-deg",
                "attitude/theta-deg",
                "attitude/phi-deg",
            ]
        elif prop == 'velocity':
            prop = [
                "velocities/v-north-fps",
                "velocities/v-east-fps",
                "velocities/v-down-fps",
            ]
        else:
            raise Exception("Property {} doesn't exist!".format(prop))

        if id == 1:
            return [
                [self.fdm1[item] for item in prop]
            ]
        elif id == 2:
            return [
                [self.fdm2[item] for item in prop]
            ]
        elif id == 0:
            return [
                [self.fdm1[item] for item in prop],
                [self.fdm2[item] for item in prop]
            ]
        else:
            raise Exception("Plane {} doesn't exist!".format(id))


    def sendAction(self, action, id):
        action_space = [
            "fcs/aileron-cmd-norm",
            "fcs/elevator-cmd-norm",
            "fcs/rudder-cmd-norm",
            "fcs/throttle-cmd-norm",
        ]
        
        if not 0 < id <= len(self.fdm):
            raise Exception("Plane {} doesn't exist!".format(id))

        for i in range(len(action_space)):
            # print("?{}".format(action))
            self.fdm[id - 1][action_space[i]] = action[0][i]

    def getDistance(self, id=None):
        positionEci = self.getProperty("positionEci", 0)
        if id == 1:
            return np.array(positionEci[1]) - np.array(positionEci[0])
        elif id == 2:
            return np.array(positionEci[0]) - np.array(positionEci[1])
        elif id is None:
            return np.linalg.norm(np.array(positionEci[1]) - np.array(positionEci[0]))
        else:
            raise Exception("Plane {} doesn't exist!".format(id))

    def getHP(self, id):
        if id == 1:
            return self.fdm1_hp
        elif id == 2:
            return self.fdm2_hp
        elif id == 0:
            return [
                self.fdm1_hp,
                self.fdm2_hp,
            ]

    def nof(self):
        return self.nof

    def damage(self):
        positionEci = self.getProperty("positionEci", 0)
        attitude = self.getProperty("attitudeRad", 0)

        dPosition_1 = np.array(positionEci[1]) - np.array(positionEci[0])
        theta_1 = np.pi / 2 - attitude[0][1]
        phi_1 = np.pi / 2 - attitude[0][0]
        heading_1 = np.array([
            np.cos(theta_1),
            np.sin(theta_1) * np.cos(phi_1),
            np.sin(theta_1) * np.sin(phi_1),
        ])

        dPosition_2 = -dPosition_1
        theta_2 = np.pi / 2 - attitude[1][1]
        phi_2 = np.pi / 2 - attitude[1][0]
        heading_2 = np.array([
            np.cos(theta_2),
            np.sin(theta_2) * np.cos(phi_2),
            np.sin(theta_2) * np.sin(phi_2),
        ])

        if 500 <= np.linalg.norm(dPosition_1) <= 3000:
            angle1 = np.arcsin(
                np.linalg.norm(np.cross(dPosition_1, heading_1)) /
                (np.linalg.norm(dPosition_1) * np.linalg.norm(heading_1))
            )

            if -1 <= angle1 / np.pi * 180 <= 1:
                self.fdm2_hp -= (3000 - np.linalg.norm(dPosition_1)) / 2500 / 120

            angle2 = np.arcsin(
                np.linalg.norm(np.cross(dPosition_2, heading_2)) /
                (np.linalg.norm(dPosition_2) * np.linalg.norm(heading_2))
            )

            if -1 <= angle2 / np.pi * 180 <= 1:
                self.fdm1_hp -= (3000 - np.linalg.norm(dPosition_2)) / 2500 / 120
    
    def terminal(self):
        if self.fdm1_hp <= 0 and self.fdm2_hp > 0:
            return 2
        if self.fdm2_hp <= 0 and self.fdm1_hp > 0:
            return 1
        if self.fdm1_hp <= 0 and self.fdm2_hp <= 0:
            return -1
        return 0
        

    def step(self, realtime=0):

        # self.sendAction(self.getAction(self.getProperty('position', 1)), 1)
        # self.sendAction(self.getAction(self.getProperty('position', 2)), 2)

        self.nof += 1

        self.fdm1.run()
        self.fdm2.run()

        self.damage()

        if self.nof >= 600:
            return 1
        
        if realtime != 0:
            time.sleep(self.fdm1.get_delta_t() / 1)

        return self.terminal()






        



























