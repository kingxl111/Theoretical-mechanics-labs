# Анимация движения системы 

import matplotlib.pyplot as pplt
from matplotlib.animation import FuncAnimation
import numpy as np

T = np.linspace(0, 10, 1000)
Psi = np.sin(-0.5*T)
fgr = pplt.figure()
plt = fgr.add_subplot(1,1,1)
plt.axis('equal')

# Радиусы большой и малой окружностей
R = 1
r = 0.2

X0 = 0
Y0 = 0
Phi = Psi + np.pi/6

# Шаблон большой окружности 
Alp = np.linspace(0, 2*np.pi, 1000)
Xc1 = R * np.cos(Alp)
Yc1 = R * np.sin(Alp)

Disk1 = plt.plot(Xc1 + X0, Yc1 + Y0)[0]

# Координаты точки A
Ax = R*np.cos(Phi[0]) - r*np.cos(Phi[0])
Ay = R*np.sin(Phi[0]) - r*np.sin(Phi[0])
OA = plt.plot([X0, Ax],[Y0, Ay])[0]

# Шаблон малой окружности 
Xc2 = r * np.cos(Alp)
Yc2 = r * np.sin(Alp)

Disk2 = plt.plot(Xc2 + Ax, Yc2 + Ay)[0]

# Шаблон спиральной пружины
Ns = 2 # число витков
r1 = 0.06
r2 = 0.2
numpnts = np.linspace(0, 1, 50*Ns + 1)
Betas = numpnts * (2*np.pi * Ns + (np.pi/2 - Phi[0]))
Xs = np.sin(Betas) * ((r2-r1)*numpnts)
Ys = np.cos(Betas) * ((r2-r1)*numpnts)

SpPruzh = plt.plot(Xs + X0, Ys + Y0)[0]

def update(i):
    Ax = R*np.cos(Phi[i]) - r*np.cos(Phi[i])
    Ay = R*np.sin(Phi[i]) - r*np.sin(Phi[i])

    # Обновление отрезка OA
    OA.set_data([X0, Ax],[Y0, Ay])

    # Обновление малой окружности
    Disk2.set_data(Xc2 + Ax, Yc2 + Ay)

    # Обновление пружины
    Betas = numpnts * (2*np.pi * Ns + (np.pi/2 - Phi[i]))
    Xs = np.sin(Betas) * ((r2-r1)*numpnts)
    Ys = np.cos(Betas) * ((r2-r1)*numpnts)
    SpPruzh.set_data(Xs + X0, Ys + Y0)
    return

anim = FuncAnimation(fgr, update, frames = len(T), interval = 1)
fgr.show()