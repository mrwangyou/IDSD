import numpy as np



# 导弹参数
V_m = 300
theta_m = 45 * np.pi / 180
fea_m = 45 * np.pi / 180

X_m = 0
Y_m = 0
Z_m = 0



# 目标参数
V_t = 100
theta_t = 0 * np.pi / 180
fea_t = 0 * np.pi / 180

X_t = 1000
Y_t = 1000
Z_t = 0


N1 = 20
N2 = 20
g = 9.8

t = 0
dt = 0.001
n = 1

R = np.sqrt(
    (X_m - X_t) ** 2 +
    5
)




