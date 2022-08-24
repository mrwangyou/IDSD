
# Dogfight Sandbox client example
# This script show how to use the network mode to controls aircrafts.
# Before starts this script, Dogfight Sandbox must be running in "Network mode"
# dogfight_client.py is the library needed to communicate with DogFight sandbox
import sys

# sys.path.append('./')

from dogfight_sandbox_hg2.network_client_example import dogfight_client as df
import time

# Print fps function, to check the network client frequency.

t = 0
t0 = 0
t1 = 0

def print_fps():
	global t, t0, t1
	t1 = time.time()
	dt = t1 - t0
	t0 = t1
	if dt > 0:
		print(str(1 / dt))

# Enter the IP and port displayed in top-left corner of DogFight screen
df.connect("10.184.65.116", 50888)

time.sleep(2)

# Get the whole planes list in DogFight scene
# returns a list that contains planes id
planes = df.get_planes_list()
print("Planes list: {}".format(str(planes)))

df.disable_log()

# Get the id of the plane you want to control
plane_id = planes[3]  # Ennemy_1

# Reset the plane at its start state
for i in planes:
	df.reset_machine(i)

# Set plane thrust level (0 to 1)
df.set_plane_thrust(plane_id, 1)

df.set_plane_thrust(planes[1], 1)

# Set client update mode ON: the scene update must be done by client network, calling "update_scene()"
df.set_client_update_mode(True)

# Wait until plane thrust = 1
df.set_renderless_mode(True)
while t < 1:
	plane_state = df.get_plane_state(plane_id)
	# Display text & vector - !!! Must be called before update_scene() !!!
	# !!! Display text & vector only works in Client Update Mode !!!
	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
	# Update frame:
	df.update_scene()
	# print_fps()
	t = plane_state["thrust_level"]


# Activate the post-combustion (increases thrust power) 
df.activate_post_combustion(plane_id)
df.activate_post_combustion(planes[2])

# Set broomstick pitch level (<0 : aircraft pitch up, >0 : aircraft pitch down)
df.set_plane_pitch(plane_id, -0.5)
df.set_plane_pitch(planes[1], -0.5)

# Wait until plane pitch attitude >= 15
p = 0
while p < 15:
	# Timer to 1/20 s.
	# As Client update mode is ON, the renderer runs to 20 FPS until pitch >= 15°
	time.sleep(1/60)
	plane_state = df.get_plane_state(plane_id)
    
	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
	df.update_scene()
	p = plane_state["pitch_attitude"]

# Reset broomstick to 0
df.stabilize_plane(plane_id)

# Retract landing gear
df.retract_gear(plane_id)

# Wait until linear speed >= 500 km/h
s = 0
while s < 1000: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
	plane_state = df.get_plane_state(plane_id)
	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
	df.update_scene()
	s = plane_state["altitude"]

# # Wait until linear speed >= 500 km/h
# s = 0
# while s < 500 / 3.6: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
# 	plane_state = df.get_plane_state(plane_id)
# 	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
# 	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
# 	df.update_scene()
# 	s = plane_state["linear_speed"]

missiles = df.get_machine_missiles_list(plane_id)
print(missiles)

# Get the missile id at slot 0
missile_slot = 0
missile_id = missiles[missile_slot]

df.fire_missile(plane_id, missile_slot)

df.set_missile_target(missile_id, 'ally_2')
df.set_missile_life_delay(missile_id, 0)

df.set_renderless_mode(False)

print(df.get_targets_list(missile_id))
raise Exception("")

# When speed is around 500 km/h, post-combustion booster is turned off
df.deactivate_post_combustion(plane_id)

# Set Renderless mode ON
df.set_renderless_mode(True)

# Wait while Renderless mode setting up:
f = False
while not f:
	f = df.get_running()["running"]

print("RenderLess running")

# Wait until plane altitude >= 5000 m
a = 0

while a < 5000:
	print_fps()
	df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
	df.update_scene()
	plane_state = df.get_plane_state(plane_id)
	a = plane_state["altitude"]

# When cruising speed & altitude are OK, setups and starts the autopilot
df.set_plane_autopilot_altitude(plane_id, 200)
df.set_plane_autopilot_heading(plane_id, 360-90)
df.set_plane_autopilot_speed(plane_id, 500 / 3.6)
df.activate_autopilot(plane_id)

# Renderless mode OFF
df.set_renderless_mode(False)

# Wait while Renderless mode setting up
f = False
while not f:
	f = df.get_running()["running"]

# Client update mode OFF
df.set_client_update_mode(False)

# Disconnect from the Dogfight server

df.disconnect()


