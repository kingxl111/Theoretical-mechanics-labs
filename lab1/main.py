import math
import sympy as s
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def Vect_arrow(X, Y, ValX, ValY):
    a = 0.2
    b = 0.3
    Arx = np.array([-b, 0, -b])
    Ary = np.array([a, 0, -a])
    alp = math.atan2(ValY, ValX)
    RotArx = Arx*np.cos(alp) - Ary*np.sin(alp)
    RotAry = Arx*np.sin(alp) + Ary*np.cos(alp)

    Arx = X + ValX + RotArx
    Ary = Y + ValY + RotAry
    return Arx, Ary

t = s.Symbol('t')

# Вариант 16
x = s.cos(1.8*t + 0.2*(s.cos(12*t)**2))*(2 + s.sin(12*t))
y = s.sin(1.8*t + 0.2*(s.cos(12*t)**2))*(2 + s.sin(12*t))

# Скорость
Vx = s.diff(x)
Vy = s.diff(y)
Vmod = s.sqrt(Vx*Vx + Vy*Vy)

# Ускорение
Ax = s.diff(Vx)
Ay = s.diff(Vy)
Amod = s.sqrt(Ax*Ax + Ay*Ay)

# Вычисление тангенциального ускорения
cos_a = (Vx*Ax + Vy*Ay)/(Vmod * Amod)
Wtaux = Vx/Vmod*Amod*cos_a
Wtauy = Vy/Vmod*Amod*cos_a
Wtau = s.diff(Vmod, t)

# Вычисление радиуса кривизны
rho = (Vmod*Vmod)/s.sqrt(Amod*Amod-Wtau*Wtau)
Rhox = -s.diff(y, t)*(s.diff(x, t)**2 + s.diff(y, t)**2)/(s.diff(x, t)*s.diff(y, t, 2) - s.diff(x, t, 2)*s.diff(y, t))
Rhoy = s.diff(x, t)*(s.diff(x, t)**2 + s.diff(y, t)**2)/(s.diff(x, t)*s.diff(y, t, 2) - s.diff(x, t, 2)*s.diff(y, t))

step = 700                   # разбиение временного отрезка
T = np.linspace(0,10,step)   # список точек разбиения

X = np.zeros_like(T)         
Y = np.zeros_like(T)   

VX = np.zeros_like(T)
VY = np.zeros_like(T)

AX = np.zeros_like(T)
AY = np.zeros_like(T)

Rho = np.zeros_like(T)

for i in np.arange(len(T)):
    print("Calculations progress:", "%.1f" % ((i / step) * 100), "%")
    X[i] = s.Subs(x, t, T[i])       # Замена: подставляем в x(t) определенное T[i] вместо t 
    Y[i] = s.Subs(y, t, T[i])

    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])

    AX[i] = s.Subs(Ax, t, T[i])
    AY[i] = s.Subs(Ay, t, T[i])
    # print(math.sqrt(AX[i]**2 + AY[i]**2))

    Rho[i] = s.Subs(rho, t, T[i])

# Создание окна
fig = plt.figure("lab1") 

# Добавление осей 
axis = fig.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim = [-10,10], ylim = [-10,10])

# Отрисовка точек
axis.plot(X,Y)

# Отрисовка начального положения тела
Pnt = axis.plot(X[0],Y[0], marker = 'o')[0]

# Отрисовка начального вектора скорости
Vp = axis.plot([X[0], (X[0]+VX[0])],[Y[0], (Y[0]+VY[0])], 'red', label = 'Вектор скорости')[0]

# Отрисовка начального вектора ускорения
Ap = axis.plot([X[0], (X[0]+AX[0])],[Y[0], (Y[0]+AY[0])], 'blue', label = 'Вектор ускорения')[0] 

RhoX = X[0] + VY[0] * Rho[0]/math.sqrt(VX[0]**2 + VY[0]**2)
RhoY = Y[0] - VX[0] * Rho[0]/math.sqrt(VX[0]**2 + VY[0]**2)

# Отрисовка начального радиуса кривизны
Rhop = axis.plot([X[0], RhoX], [Y[0], RhoY], 'black', label = 'Вектор радиуса кривизны')[0]

# Отрисовка начального радиус вектора
RLine = axis.plot([0, X[0]], [0, Y[0]], 'orange', label = 'Радиус-вектор')[0]

# Отрисовка стрелок для всех векторов 
RAx1, RAy1 = Vect_arrow(X[0], Y[0], VX[0], VY[0])
Varrow = axis.plot(RAx1, RAy1, 'red')[0]

RAx2, RAy2 = Vect_arrow(X[0], Y[0], AX[0], AY[0])
Aarrow = axis.plot(RAx2, RAy2, 'blue')[0]

RAx3, RAy3 = Vect_arrow(X[0], Y[0], RhoX, RhoY)
RHarrow = axis.plot(RAx3, RAy3, 'black')[0]

RAx4, RAy4 = Vect_arrow(0, 0, X[0], Y[0])
Rarrow = axis.plot(RAx4, RAy4, 'orange')[0]

def anim(i):
    RhoX = X[i] + VY[i] * Rho[i]/math.sqrt(VX[i]**2 + VY[i]**2)
    RhoY = Y[i] - VX[i] * Rho[i]/math.sqrt(VX[i]**2 + VY[i]**2)

    Pnt.set_data(X[i], Y[i])

    Vp.set_data([X[i], (X[i]+VX[i])],[Y[i], (Y[i]+VY[i])])

    Ap.set_data([X[i], (X[i]+AX[i])],[Y[i], (Y[i]+AY[i])])
    RAx2, RAy2 = Vect_arrow(X[i], Y[i], AX[i], AY[i])
    Varrow.set_data(RAx2, RAy2)

    Rhop.set_data([X[i], RhoX], [Y[i], RhoY])
    RLine.set_data([0, X[i]], [0, Y[i]])

    RAx1, RAy1 = Vect_arrow(X[i], Y[i], VX[i], VY[i])
    Varrow.set_data(RAx1, RAy1)

    RAx2, RAy2 = Vect_arrow(X[i], Y[i], AX[i], AY[i])
    Aarrow.set_data(RAx2, RAy2)

    RAx3, RAy3 = Vect_arrow(X[i], Y[i], RhoX-X[i], RhoY-Y[i])
    RHarrow.set_data(RAx3, RAy3)

    RAx4, RAy4 = Vect_arrow(0, 0, X[i], Y[i])
    Rarrow.set_data(RAx4, RAy4)

    number = int(input())

an = FuncAnimation(fig, anim, frames = step, interval = 1)

fig.show()
