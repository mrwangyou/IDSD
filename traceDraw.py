from matplotlib import pyplot as plt
 
fig = plt.figure()
ax1 = plt.axes(projection='3d')

import numpy as np

f = open('./log/tracelog.txt', 'r', encoding="UTF-8")

flag = 0
for line in f:
    if flag == 0:
        x0 = float(line.split()[0])
        y0 = float(line.split()[1])
        z0 = float(line.split()[2])
        x = np.array(0)
        y = np.array(0)
        z = np.array(0)
        a0 = float(line.split()[3])
        b0 = float(line.split()[4])
        c0 = float(line.split()[5])
        p = np.array(a0 - x0)
        q = np.array(b0 - y0)
        r = np.array(c0 - z0)
        flag = 1
    x = np.append(x, np.array(float(line.split()[0]) - x0))
    y = np.append(y, np.array(float(line.split()[1]) - y0))
    z = np.append(z, np.array(float(line.split()[2]) - z0))
    p = np.append(p, np.array(float(line.split()[3]) - x0))
    q = np.append(q, np.array(float(line.split()[4]) - y0))
    r = np.append(r, np.array(float(line.split()[5]) - z0))
# 基于ax1变量绘制三维图
#设置xyz方向的变量（空间曲线）
# z=np.linspace(0,13,1000)#在1~13之间等间隔取1000个点
# x=5*np.sin(z)
# y=5*np.cos(z)
 
# #设置xyz方向的变量（散点图）
# zd=13*np.random.random(100)
# xd=5*np.sin(zd)
# yd=5*np.cos(zd)
 
#设置坐标轴
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
# ax1.scatter(xd,yd,zd,cmap='Blues') #绘制散点图
ax1.plot3D(y, z, x, 'red')  # 绘制空间曲线
ax1.plot3D(q, r, p, 'blue')  # 绘制空间曲线
plt.show()#显示图像

