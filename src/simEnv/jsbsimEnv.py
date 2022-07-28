import math
import os
import random
import sys
import time
from turtle import position

import jsbsim
import numpy as np

sys.path.append(str(jsbsim.get_default_root_dir()) + '/pFCM/')


class JsbsimEnv():

    def __init__(
        self,
        fdm_id       = 1,
        fdm_aircraft = 'f16',    
        fdm_ic_v     = 500,      # Calibrated Velocity (knots) https://skybrary.aero/articles/calibrated-airspeed-cas#:~:text=Definition,port%20caused%20by%20airflow%20disruption).
        fdm_ic_lat   = 0,        # Latitude (degree) 纬度
        fdm_ic_long  = 0,        # Longitude (degree) 经度
        fdm_ic_h     = 30005.5,  # Height above sea level (feet)
        fdm_ic_psi   = 0,        # Yaw 偏航角，绕Z轴，JSBSim将按照这里列出的顺序进行欧拉角计算
        fdm_ic_theta = 0,        # Pitch 俯仰角，绕Y轴
        fdm_ic_phi   = 0,        # Roll 翻滚角，绕X轴

        fdm_hp       = 1,
        fdm_fgfs     = False,
        flight_mode   = 1,  # 0 for flight test
    ) -> None:

        # FDM Initialization
        self.fdm = jsbsim.FGFDMExec(None)
        
        # Aircraft Loading
        self.fdm.load_model(fdm_aircraft)  
        
        # Visualization fgfs Port 5550
        if fdm_fgfs is True:
            self.fdm.set_output_directive('./data_output/flightgear{}.xml'.format(fdm_id))  
        
        # Velocity Initialization
        self.fdm['ic/vc-kts'] = fdm_ic_v  

        # Position Initialization
        self.fdm["ic/lat-gc-deg"] = fdm_ic_lat  
        self.fdm["ic/long-gc-deg"] = fdm_ic_long  
        self.fdm["ic/h-sl-ft"] = fdm_ic_h  
        self.fdm["ic/psi-true-deg"] = fdm_ic_psi
        self.fdm["ic/theta-deg"] = fdm_ic_theta
        self.fdm["ic/phi-deg"] = fdm_ic_phi

        ##########################
        ## Model Initialization ##
        self.fdm.run_ic()       ##
        ##########################

        # Engine Turning on
        self.fdm["propulsion/starter_cmd"] = 1

        # Refueling
        if flight_mode == 0:
            self.fdm["propulsion/refuel"] = 1

        # First but not Initial
        self.fdm.run()
        self.fdm["propulsion/active_engine"] = True
        self.fdm["propulsion/set-running"] = 0

        self.nof = 1  # Number of frames

        # HP setting
        self.fdm_hp = fdm_hp

    def getProperty(
        self,
        prop,
    ) -> list:
        if prop == 'position':
            print(self.fdm["position/lat-gc-deg"])
            prop = [
                "position/lat-gc-deg",  # Latitude 纬度
                "position/long-gc-deg",  # Longitude 经度
                "position/h-sl-ft",  # Altitude above sea level 海拔
            ]
        elif prop == 'positionEci':  # Earth-centered inertial
            prop = [
                "position/eci-x-ft",  # 指向经纬度为0的点
                "position/eci-y-ft",  # 左手系决定
                "position/eci-z-ft",  # 指向北极点
            ]
        elif prop == 'positionEcef':
            prop = [
                "position/ecef-x-ft",
                "position/ecef-y-ft",
                "position/ecef-z-ft",
            ]
        elif prop == 'attitudeRad':
            prop = [
                "attitude/psi-rad",  # Yaw 偏航角
                "attitude/theta-rad",  # Pitch 俯仰角
                "attitude/phi-rad",  # Roll 翻滚角
            ]
        elif prop == 'attitudeDeg':
            prop = [
                "attitude/psi-deg",
                "attitude/theta-deg",
                "attitude/phi-deg",
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
            return self.fdm[prop]

        return [
            self.fdm[item] for item in prop
        ]

    def sendAction(
        self,
        action,
    ):
        action_space = [
            "fcs/aileron-cmd-norm",
            "fcs/elevator-cmd-norm",
            "fcs/rudder-cmd-norm",
            "fcs/throttle-cmd-norm",
        ]

        for i in range(len(action_space)):
            self.fdm[action_space[i]] = action[0][i]

    def getHP(self):  # Health point
        return self.fdm_hp
    
    def damage(self, dmg):
        self.fdm_hp = self.fdm_hp - dmg

    def getNof(self):  # Number of frames
        return self.nof
    
    def step(self):
        self.nof = self.nof + 1
        self.fdm.run()

    def terminate(self):  # Unused
        if self.fdm_hp <= 0:
            return 1
        else:
            return 0


class DogfightEnv():

    def __init__(
        self,
    ) -> None:
        self.fdm = [
            JsbsimEnv(
                fdm_id=1,
                fdm_fgfs=False,
            ),

            JsbsimEnv(
                fdm_id=2,
                fdm_aircraft='f16_1',
                fdm_ic_lat=0.005,
                fdm_ic_psi=180,
                fdm_fgfs=False,
            ),
        ]
        self.file = open('./log/tracelog.txt', 'w', encoding='UTF-8')

    def getFdm(
        self,
        fdmId=0,
    ):
        if fdmId == 0:  # Caution! 
            return self.fdm
        else:
            return self.fdm[fdmId - 1]

    def getDistanceVector(self, ego):
        positionEci1 = self.getFdm(1).getProperty("positionEci")  # A list of size [3]
        positionEci2 = self.getFdm(2).getProperty("positionEci")  # A list of size [3]
        # print('***{}'.format(np.array(positionEci1)))
        # print('***{}'.format(np.array(positionEci2)))
        # if np.isnan(positionEci1[0]):
        #     raise Exception(positionEci1, np.array(positionEci1))
        if ego == 1:
            return np.array(positionEci2) - np.array(positionEci1)
        elif ego == 2:
            return np.array(positionEci1) - np.array(positionEci2)
        else:
            raise Exception("Plane {} doesn\'t exist".format(ego))

    def getDistance(self):
        return np.linalg.norm(self.getDistanceVector(1))
    
    def getHP(self):
        return [
            self.getFdm(1).getHP(),
            self.getFdm(2).getHP(),
        ]

    def damage(self):
        attitude1 = self.getFdm(1).getProperty("attitudeRad")  # A list of size [3]
        attitude2 = self.getFdm(2).getProperty("attitudeRad")  # A list of size [3]

        theta_1 = np.pi / 2 - attitude1[1]
        psi_1 = np.pi / 2 - attitude1[0]
        heading_1 = np.array([
            np.cos(theta_1),
            np.sin(theta_1) * np.cos(psi_1),
            np.sin(theta_1) * np.sin(psi_1),
        ])

        theta_2 = np.pi / 2 - attitude2[1]
        psi_2 = np.pi / 2 - attitude2[0]
        heading_2 = np.array([
            np.cos(theta_2),
            np.sin(theta_2) * np.cos(psi_2),
            np.sin(theta_2) * np.sin(psi_2),
        ])

        if 500 <= self.getDistance() <= 3000:
            # angle1 = np.arcsin(
            #     np.linalg.norm(np.cross(self.getDistanceVector(ego=1), heading_1)) /
            #     (self.getDistance() * np.linalg.norm(heading_1))
            # )

            angle1 = np.arccos(
                np.dot(self.getDistanceVector(ego=1), heading_1) / 
                (self.getDistance() * np.linalg.norm(heading_1))
            )

            if -1 <= angle1 / np.pi * 180 <= 1:
                self.getFdm(2).damage((3000 - self.getDistance()) / 2500 / 120)

            # angle2 = np.arcsin(
            #     np.linalg.norm(np.cross(self.getDistanceVector(ego=2), heading_2)) /
            #     (self.getDistance() * np.linalg.norm(heading_2))
            # )

            angle2 = np.arccos(
                np.dot(self.getDistanceVector(ego=2), heading_2) / 
                (self.getDistance() * np.linalg.norm(heading_2))
            )

            if -1 <= angle2 / np.pi * 180 <= 1:
                self.getFdm(1).damage((3000 - self.getDistance()) / 2500 / 120)
        

    def terminate(self):
        if self.getFdm(1).getHP() <= 0 and self.getFdm(2).getHP() > 0:
            self.file.close()
            return 2
        elif self.getFdm(2).getHP() <= 0 and self.getFdm(1).getHP() > 0:
            self.file.close()
            return 1
        elif self.getFdm(1).getHP() <= 0 and self.getFdm(2).getHP() <= 0:
            self.file.close()
            return -1
        else:
            return 0

    def getNof(self):
        if self.getFdm(1).getNof() != self.getFdm(2).getNof():
            raise Exception("FDM NoF Error!", self.getFdm(1).getNof(), self.getFdm(2).getNof())
        return self.getFdm(1).getNof()

    def step(self, playSpeed=0):
        
        print("nof: {}".format(self.getNof()))
        print("*Distance: \t{}".format(self.getDistance()))
        print("**HP: {0[0]}\t\t{0[1]}\n".format(self.getHP()))

        self.getFdm(1).step()
        self.getFdm(2).step()

        self.damage()

        if self.getNof() >= 1000:
            return -1

        self.file.write("{} {} {} {} {} {}\n".format(
            self.getFdm(1).getProperty("positionEcef")[0],
            self.getFdm(1).getProperty("positionEcef")[1],
            self.getFdm(1).getProperty("positionEcef")[2],
            self.getFdm(2).getProperty("positionEcef")[0],
            self.getFdm(2).getProperty("positionEcef")[1],
            self.getFdm(2).getProperty("positionEcef")[2],
        ))

        if playSpeed != 0:
            time.sleep(self.getFdm(1).get_delta_t() / playSpeed)

        return self.terminate()































