import matplotlib.pyplot as p
from matplotlib.animation import FuncAnimation
import numpy as np

from scipy.integrate import odeint

def SystDiffEq(y, t, g, m1, m2, m3, R, r, M1, M2):
    # начальные условия
    # y = [phi, theta, phi', theta'] -> dy = [phi', theta', phi'', theta'']
    dy = np.zeros_like(y)

    # У нас уже есть производные первого порядка
    dy[0] = y[2]
    dy[1] = y[3]

    # a11 * phi'' + a12 * theta'' = b1
    # a21 * phi'' + a22 * theta'' = b2

    # Далее пользуемся правилом Крамера
    
    # Коэффициент перед phi'' в первом уравнении системы
    a11 = (3*m2 + 2*m3/3)*(R-r)*(R-r)
    # Коэффициент перед theta'' в первом уравнении системы
    a12 = m2*R*(R-r)
    # Правая часть первого уравнения
    b1 = -2*c*y[0] + 2*M2 + (2*m2+m3)*g*np.sin(y[0])*(R-r)

    # Коэффициент перед phi'' во втором уравнении системы
    a21 = m2*(R-r)*R
    # Коэффициент перед theta'' во втором уравнении системы
    a22 = (m1+m2)*R*R
    # Правая часть второго уравнения
    b2  = 2*M1

    # Непосредственно вторые производные
    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)
    
    return dy


# Масса в кг
m1 = 2 
m2 = 1
m3 = 1

# Радиусы дисков
R = 1
r = 0.2

c = 2
M1 = 0.5
M2 = 0.2

g = 9.81

T = np.linspace(0, 10, 100)

# По условию: 
y0 = [np.pi/6, np.pi/2, 0, 2]

Y = odeint(SystDiffEq, y0, T, (g, m1, m2, m3, R, r, M1, M2))

# Y по своей сути - матрица,  столбцы которой имеют вид:
# [phi theta phi' theta'] 

# Присваиваем столбцы матрицы соответствующим координатам
Phi = Y[:,0]
Theta = Y[:,1] # n.sin(0.5*T) + 1.1
Phit = Y[:,2]
Thetat = Y[:,3]

# Количество графиков
graphics_count = 5

fgrp = p.figure()
plPhi = fgrp.add_subplot(graphics_count,1,1)
plPhi.plot(T, Phi)

plTheta = fgrp.add_subplot(graphics_count,1,2)
plTheta.plot(T, Theta)

Phitt = np.zeros_like(T)
Thetatt = np.zeros_like(T)

for i in range(len(T)):
    Phitt[i] = SystDiffEq(Y[i], T[i], g, m1, m2, m3, R, r, M1, M2)[2]
    Thetatt[i] = SystDiffEq(Y[i], T[i], g, m1, m2, m3, R, r, M1, M2)[3]


Nax = m2*(R-r)*(Phitt*np.cos(Phi) - (Phit)*(Phit)*np.sin(Phi))
Nay = m2*((r-R)*(Phitt*np.sin(Phi) + (Phit)*(Phit)*np.cos(Phi)) + g)

Fc = m2*(R*Thetatt + (R-r)*Phitt) / 2

plNax = fgrp.add_subplot(graphics_count, 1, 3)
plNax.plot(T, Nax)

plNay = fgrp.add_subplot(graphics_count, 1, 4)
plNay.plot(T, Nay)

plFc = fgrp.add_subplot(graphics_count, 1, 5)
plFc.plot(T, Fc)

p.show()

# # # ======= анимация системы =========
fgr = p.figure()
plt = fgr.add_subplot(1,1,1)
plt.axis('equal')

# Радиусы большой и малой окружностей
R = 1
r = 0.2

X0 = 0
Y0 = 0
# Phi = np.linspace(0, -100*np.pi, 1000)

# Шаблон большой окружности 
Alp = np.linspace(0, 1000*np.pi, 100)
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
p.show()






