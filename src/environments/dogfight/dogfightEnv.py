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
		
		try:
			df.get_planes_list()
		except:
			print('Run for the first time.')
			df.connect(host, int(port))

		time.sleep(2)

		planes = df.get_planes_list()
		# print("Planes list: {}".format(str(planes)))

		df.disable_log()

		self.planeID = planes[1]

		# print(df.get_plane_state(self.planeID))
		# warnings.warn('Dogfight simulation environments have no global data!')

		for i in planes:
			df.reset_machine(i)

		# Set plane thrust level (0 to 1)
		df.set_plane_thrust(planes[3], 1)

		df.set_plane_thrust(planes[1], 1)

		# Set client update mode ON: the scene update must be done by client network, calling "update_scene()"
		df.set_client_update_mode(True)

		# Wait until plane thrust = 1
		df.set_renderless_mode(True)

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
			# print(df.get_plane_state(self.planeID)['horizontal_speed'])
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

		

		# Wait until linear speed >= 500 km/h
		s = 0
		while s < 1000: # Linear speed is given in m/s. To translate in km/h, just divide it by 3.6
			plane_state = df.get_plane_state(planes[3])
			df.update_scene()
			s = plane_state["altitude"]
			# print(df.get_plane_state(self.planeID)['horizontal_speed'])
			# print((df.get_plane_state(self.planeID)['horizontal_speed'] ** 2 + df.get_plane_state(self.planeID)['vertical_speed'] ** 2) / (df.get_plane_state(self.planeID)['linear_speed'] ** 2))

		df.set_plane_yaw(self.planeID, 1)

		missiles = df.get_machine_missiles_list(planes[3])
		# print(missiles)

		# Get the missile id at slot 0
		missile_slot = 1
		missile_id = missiles[missile_slot]

		df.fire_missile(planes[3], missile_slot)

		df.set_missile_target(missile_id, 'ally_2')
		df.set_missile_life_delay(missile_id, 40)

		# df.set_renderless_mode(False)
		
		# while True:
		# 	self.step()

	def getProperty(
		self,
		prop,
	) -> list:
		if prop == 'position':
			return [
				df.get_plane_state(self.planeID)['position'][0],
				df.get_plane_state(self.planeID)['position'][2],
				df.get_plane_state(self.planeID)['position'][1],
			]
		elif prop == 'positionEci':
			warnings.warn('Dogfight simulation environments have no global data!')
			return [
				df.get_plane_state(self.planeID)['position'][0],
				df.get_plane_state(self.planeID)['position'][2],
				df.get_plane_state(self.planeID)['position'][1],
			]
		elif prop == 'positionEcef':
			warnings.warn('Dogfight simulation environments have no global data!')
			return [
				df.get_plane_state(self.planeID)['position'][0],
				df.get_plane_state(self.planeID)['position'][2],
				df.get_plane_state(self.planeID)['position'][1],
			]
		elif prop == 'attitudeRad':
			return [
				df.get_plane_state(self.planeID)['heading'] / 180 * np.pi,
				df.get_plane_state(self.planeID)['pitch_attitude'] / 180 * np.pi,
				df.get_plane_state(self.planeID)['roll_attitude'] / 180 * np.pi,
			]
		elif prop == 'attitudeDeg':
			return [
				df.get_plane_state(self.planeID)['heading'],
				df.get_plane_state(self.planeID)['pitch_attitude'],
				df.get_plane_state(self.planeID)['roll_attitude'],
			]
		elif prop == 'pose':
			return [
				df.get_plane_state(self.planeID)['position'][0],
				df.get_plane_state(self.planeID)['position'][2],
				df.get_plane_state(self.planeID)['position'][1],
				df.get_plane_state(self.planeID)['heading'],
				df.get_plane_state(self.planeID)['pitch_attitude'],
				df.get_plane_state(self.planeID)['roll_attitude'],
			]
		elif prop == 'velocity':
			warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
			return [
				df.get_plane_state(self.planeID)['horizontal_speed'],
				df.get_plane_state(self.planeID)['linear_speed'],
				-df.get_plane_state(self.planeID)['vertical_speed'],
			]
		else:
			raise Exception("Property {} doesn't exist!".format(prop))


	def getHP(self):
		return df.get_health(self.planeID)['health_level']

	def sendAction(
		self,
		action,  # [thrust, brake, flaps, pitch, roll, yaw]
		actionType=None,
	):
		if actionType == None:
			# df.set_plane_thrust(self.planeID, action[0])
			df.set_plane_brake(self.planeID, action[1])
			df.set_plane_flaps(self.planeID, action[2])
			df.set_plane_pitch(self.planeID, action[3])
			df.set_plane_roll(self.planeID, action[4])
			df.set_plane_yaw(self.planeID, action[5])
		elif actionType == 'thrust' or actionType == 'Thrust':
			df.set_plane_thrust(self.planeID, action)
		elif actionType == 'brake' or actionType == 'Brake':
			df.set_plane_brake(self.planeID, action)
		elif actionType == 'flaps' or actionType == 'Flaps':
			df.set_plane_flaps(self.planeID, action)
		elif actionType == 'pitch' or actionType == 'Pitch':
			df.set_plane_pitch(self.planeID, action)
		elif actionType == 'roll' or actionType == 'Roll':
			df.set_plane_roll(self.planeID, action)
		elif actionType == 'yaw' or actionType == 'Yaw':
			df.set_plane_yaw(self.planeID, action)
		else:
			raise Exception("Action {} doesn't exist!".format(actionType))

	def step(
		self,
	):
		df.update_scene()


if __name__ == '__main__':
	args = parse_args()

	win = 0
	summ = 0

	d = DogfightEnv(
		args.host,
		args.port,
	)
	while True:
		d.sendAction([random.random() * 2 - 1] * 6)
		d.step()
		# print(" {0[destroyed]} {0[wreck]} {0[crashed]} {0[active]}".format(df.get_missile_state('Meteorennemy_2.1')))
		if not df.get_missile_state('Meteorennemy_2.1')['active']:
			summ += 1
			if d.getHP() >= .9:
				win += 1
			f = open('./random.txt', 'a')
			f.write("{} / {}\n".format(win, summ))
			f.close()
			d.__init__(
				args.host,
				args.port,
			)
