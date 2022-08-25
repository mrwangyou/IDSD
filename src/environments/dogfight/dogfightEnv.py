import argparse
import sys
import time

import jsbsim

sys.path.append(str(jsbsim.get_default_root_dir()) + '/pFCM/')

from src.environments.dogfight.dogfight_sandbox_hg2.network_client_example import \
    dogfight_client as df

# Print fps function, to check the network client frequency.


def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--host', default='10.184.0.0', metavar='str', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', metavar='str', help='specifies Harfang port id')
    args = parser.parse_args()
    return args


class DogfightEnv():

	def __init__(
		self,
		host='10.184.0.0',
		port='50888',
	) -> None:
		
		# Enter the IP and port displayed in top-left corner of DogFight screen
		df.connect(host, int(port))

		time.sleep(2)

		planes = df.get_planes_list()
		print("Planes list: {}".format(str(planes)))

		df.disable_log()

		for i in planes:
			df.reset_machine(i)

		# Set plane thrust level (0 to 1)
		df.set_plane_thrust(planes[3], 1)

		df.set_plane_thrust(planes[1], 1)

		# Set client update mode ON: the scene update must be done by client network, calling "update_scene()"
		df.set_client_update_mode(True)

		# Wait until plane thrust = 1
		# df.set_renderless_mode(True)

		t = 0
		while t < 1:
			plane_state = df.get_plane_state(planes[3])
			df.update_scene()
			t = plane_state["thrust_level"]


		# Activate the post-combustion (increases thrust power) 
		# 如果启用了Renderless mode，那么可能打开了，但是最终的渲染没有显示
		df.activate_post_combustion(planes[3])
		df.activate_post_combustion(planes[1])

		# Set broomstick pitch level (<0 : aircraft pitch up, >0 : aircraft pitch down)
		df.set_plane_pitch(planes[3], -0.5)
		df.set_plane_pitch(planes[1], -0.5)

		# Wait until plane pitch attitude >= 15
		p = 0
		while p < 15:
			time.sleep(1/60)
			plane_state = df.get_plane_state(planes[3])
			df.update_scene()
			p = plane_state["pitch_attitude"]

		# Reset broomstick to 0
		df.stabilize_plane(planes[3])
		df.stabilize_plane(planes[1])

		# Retract landing gear
		df.retract_gear(planes[3])
		df.retract_gear(planes[1])

		# Wait until linear speed >= 500 km/h
		s = 0
		while s < 1000: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
			plane_state = df.get_plane_state(planes[3])
			df.update_scene()
			s = plane_state["altitude"]

		# # Wait until linear speed >= 500 km/h
		# s = 0
		# while s < 500 / 3.6: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
		# 	plane_state = df.get_plane_state(planes[3])
		# 	# df.display_2DText([0.25, 0.75], "Plane speed: " + str(plane_state["linear_speed"]), 0.04, [1, 0.5, 0, 1])
		# 	# df.display_vector(plane_state["position"], plane_state["move_vector"], "Linear speed: " + str(plane_state["linear_speed"]), [0, 0.02], [0, 1, 0, 1], 0.02)
		# 	df.update_scene()
		# 	s = plane_state["linear_speed"]

		missiles = df.get_machine_missiles_list(planes[3])
		print(missiles)

		# Get the missile id at slot 0
		missile_slot = 1
		missile_id = missiles[missile_slot]

		df.fire_missile(planes[3], missile_slot)

		df.set_missile_target(missile_id, 'ally_2')
		df.set_missile_life_delay(missile_id, 60)

		df.set_renderless_mode(False)

		# print(df.get_targets_list(missile_id))
		raise Exception("")



if __name__ == '__main__':
	args = parse_args()

	DogfightEnv(
		args.host,
		args.port,
	)

