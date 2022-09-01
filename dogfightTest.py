import argparse
import random
import sys
import time
import warnings

import jsbsim
import numpy as np

sys.path.append(str(jsbsim.get_default_root_dir()) + '/pFCM/')

from src.environments.dogfight.dogfight_sandbox_hg2.network_client_example import \
    dogfight_client as df

def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--host', default='10.184.0.0', metavar='str', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', metavar='str', help='specifies Harfang port id')
    args = parser.parse_args()
    return args

args = parse_args()


try:
    df.get_planes_list()
except:
    print('Run for the first time.')
    df.connect(args.host, int(args.port))
    time.sleep(2)

planes = df.get_planes_list()
# print("Planes list: {}".format(str(planes)))

df.disable_log()

planeID = planes[1]

for i in planes:
    df.reset_machine(i)

# Set plane thrust level (0 to 1)
df.set_plane_thrust(planes[3], 1)
df.set_plane_thrust(planes[1], 1)

df.set_client_update_mode(True)

# df.set_renderless_mode(True)

t = 0
while t < 1:
    plane_state = df.get_plane_state(planes[3])
    df.update_scene()
    t = plane_state["thrust_level"]


# Activate the post-combustion (increases thrust power) 
# 如果启用了Renderless mode，那么加速可能打开了，但是最终的渲染没有显示
df.activate_post_combustion(planes[3])
df.activate_post_combustion(planes[1])

# Set broomstick pitch level (<0 : aircraft pitch up, >0 : aircraft pitch down)
df.set_plane_pitch(planes[3], -0.5)
df.set_plane_pitch(planes[1], -0.5)

# Wait until plane pitch attitude >= 15
p = 0
while p < 15:
    # print(df.get_plane_state(planeID)['horizontal_speed'])
    # time.sleep(1/60)
    plane_state = df.get_plane_state(planes[3])
    df.update_scene()
    p = plane_state["pitch_attitude"]

# Reset broomstick to 0
df.stabilize_plane(planes[3])
df.stabilize_plane(planes[1])

# Retract landing gear
df.retract_gear(planes[3])
df.retract_gear(planes[1])


s = 0
while s < 1000: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
    plane_state = df.get_plane_state(planes[3])
    df.update_scene()
    s = plane_state["altitude"]

df.set_plane_yaw(planeID, 1)

missiles = df.get_machine_missiles_list(planes[3])
# print(missiles)

# Get the missile id at slot 0
missile_slot = 1
missileID = missiles[missile_slot]

df.fire_missile(planes[3], missile_slot)

df.set_missile_target(missileID, 'ally_2')
df.set_missile_life_delay(missileID, 40)

print(df.get_missile_state(missileID))

while True:
    df.update_scene()
    print(
        ((df.get_plane_state(planeID)['position'][0] - df.get_missile_state(missileID)['position'][0]) ** 2 +\
        (df.get_plane_state(planeID)['position'][1] - df.get_missile_state(missileID)['position'][1]) ** 2 +\
        (df.get_plane_state(planeID)['position'][2] - df.get_missile_state(missileID)['position'][2]) ** 2) ** .5
    )




