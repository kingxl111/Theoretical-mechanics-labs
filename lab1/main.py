import math
import sympy as s
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

t = s.Symbol('t')

# Variant 16
x = s.cos(1.8*t + 0.2*(s.cos(12*t)**2))*(2 + s.sin(12*t))
y = s.sin(1.8*t + 0.2*(s.cos(12*t)**2))*(2 + s.sin(12*t))

# Velocity
Vx = s.diff(x)
Vy = s.diff(y)

# Acceleration
Ax = s.diff(Vx)
Ay = s.diff(Vy)

step = 1351                  # time interval partition
T = np.linspace(0,10,step)   # array of partition points from 0 to 10

X = np.zeros_like(T)         # array of zeros of length like T
Y = np.zeros_like(T)   

VX = np.zeros_like(T)
VY = np.zeros_like(T)

AX = np.zeros_like(T)
AY = np.zeros_like(T)

for i in np.arange(len(T)):

    X[i] = s.Subs(x, t, T[i])       # Replacing: substitute in x(t) a certain T[i] instead of t 
    Y[i] = s.Subs(y, t, T[i])

    VX[i] = s.Subs(Vx, t, T[i])
    VY[i] = s.Subs(Vy, t, T[i])

    AX[i] = s.Subs(Ax, t, T[i])
    AY[i] = s.Subs(Ay, t, T[i])


# Creating the window
fig = plt.figure("lab1") 

# add axes
axis = fig.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim = [-4,4],ylim = [-4, 4])

# draw all points
axis.plot(X,Y)

#draw first point
Pnt = axis.plot(X[0],Y[0], marker = 'o')[0]

#draw first point velocity vector
Vp = axis.plot([X[0], X[0]+VX[0]],[Y[0], Y[0]+VY[0]], 'red')[0]

#draw first point acceleration vector
Ap = axis.plot([X[0], X[0]+AX[0]],[Y[0], Y[0]+AY[0]], 'blue')[0] 

# def Vect_arrow(X, Y, ValX, ValY):
#     a = 0.2
#     b = 0.3
#     Arx = np.array([-b, 0, -b])
#     Ary = np.array([a, 0, -a])
#     alp = math.atan2(ValY, ValX)
#     RotArx = Arx*np.cos(alp) - Ary*np.sin(alp)
#     RotAry = Arx*np.sin(alp) + Ary*np.cos(alp)

#     Arx = X + ValX + RotArx
#     Ary = Y + ValY + RotAry
#     return Arx, Ary

# RAx, RAy = Vect_arrow(X[0], Y[0], VX[0], VY[0])
# Varrow = axis.plot(RAx, RAy, 'red')[0]

def anim(i):
    Pnt.set_data(X[i], Y[i])
    Vp.set_data([X[i], X[i]+VX[i]],[Y[i], Y[i]+VY[i]])
    Ap.set_data([X[i], X[i]+AX[i]],[Y[i], Y[i]+AY[i]])
    # RAx, RAy = Vect_arrow(X[i], Y[i], VX[i], VY[i])
    # Varrow.set_data(RAx, RAy)

an = FuncAnimation(fig, anim, frames = step, interval = 1)

fig.show()
# inp = input()   
