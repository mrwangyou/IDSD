import numpy as np


def R_rp(
    env,
    id
):
    reward = 0
    attitude1 = env.getFdm(1).getProperty("attitudeRad")  # A list of size [3]
    attitude2 = env.getFdm(2).getProperty("attitudeRad")  # A list of size [3]
    
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

    angle1 = np.arccos(
        np.dot(env.getDistanceVector(ego=1), heading_1) / 
        (env.getDistance() * np.linalg.norm(heading_1))
    ) / np.pi * 180

    angle2 = np.arccos(
        np.dot(env.getDistanceVector(ego=2), heading_2) / 
        (env.getDistance() * np.linalg.norm(heading_2))
    ) / np.pi * 180
    print("Angle: {}\t{}".format(angle1, angle2))
    reward = reward + -np.abs(angle1) / 180 + 1
    reward = reward - -np.abs(angle2) / 180 - 1

    if id == 2:
        reward = -reward
    
    return 2 * reward

def R_closure(
    env,
    id
):
    reward = 0
    attitude1 = env.getFdm(1).getProperty("attitudeRad")  # A list of size [3]
    attitude2 = env.getFdm(2).getProperty("attitudeRad")  # A list of size [3]
    
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

    angle1 = np.arccos(  # 0 ~ +180
        np.dot(env.getDistanceVector(ego=1), heading_1) / 
        (env.getDistance() * np.linalg.norm(heading_1))
    ) / np.pi * 180

    angle2 = np.arccos(
        np.dot(env.getDistanceVector(ego=2), heading_2) / 
        (env.getDistance() * np.linalg.norm(heading_2))
    ) / np.pi * 180

    if angle1 <= 90 :
        reward = reward + (3000 - env.getDistance()) / 2500 / 120
    if angle2 <= 90:
        reward = reward - (3000 - env.getDistance()) / 2500 / 120
    
    if id == 2:
        reward = -reward

    return reward

def R_gunsnap(
    env,
    id
):
    reward = 0
    attitude1 = env.getFdm(1).getProperty("attitudeRad")  # A list of size [3]
    attitude2 = env.getFdm(2).getProperty("attitudeRad")  # A list of size [3]
    
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

    angle1 = np.arccos(  # 0 ~ +180
        np.dot(env.getDistanceVector(ego=1), heading_1) / 
        (env.getDistance() * np.linalg.norm(heading_1))
    ) / np.pi * 180

    angle2 = np.arccos(
        np.dot(env.getDistanceVector(ego=2), heading_2) / 
        (env.getDistance() * np.linalg.norm(heading_2))
    ) / np.pi * 180

    reward = reward + max(0, 1 / (1 + np.exp(5 * angle1 - 10)))
    reward = reward - max(0, 1 / (1 + np.exp(5 * angle2 - 10)))

    if id == 2:
        reward = -reward

    return reward

def R_deck(
    env,
    id
):
    reward = 0
    altitude1 = env.getFdm(1).getProperty("position")[2]  # A list of size [3]
    altitude2 = env.getFdm(2).getProperty("position")[2]  # A list of size [3]

    if id == 1 and altitude1 <= 10000:
        reward = -(10000 - altitude1) / 100
    if id == 2 and altitude2 <= 10000:
        reward = -(10000 - altitude2) / 100
    try:
        return max(reward, -10)
    except:
        raise Exception(reward)

def R_too_close(
    env,
    id
):
    reward = 0
    distance = env.getDistance()

    if id == 1 and distance <= 500:
        reward = -(500 - distance) / 500
    if id == 2 and distance <= 500:
        reward = -(500 - distance) / 500

    return max(reward, -10)

def R_win(
    env,
    id
):
    if env.getFdm(id^3).getHP() <= 0:
        return 100
    return 0

