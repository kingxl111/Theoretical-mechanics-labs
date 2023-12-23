import matplotlib.pyplot as p
from matplotlib.animation import FuncAnimation
import numpy as n

from scipy.integrate import odeint


def SystDiffEq(y, t, J,m,m1,l,r,alph,c,g):
    # y = [phi, psi, phi', psi'] -> dy = [phi', psi', phi'', psi'']
    dy = n.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    # a11 * phi'' + a12 * psi'' = b1
    # a21 * phi'' + a22 * psi'' = b2

    a11 = J + (m/3 + m1) * l**2 * n.sin(y[1])**2 + m1/4 * r**2
    a12 = 0
    b1 = -(m/3 + m1) * l**2 * y[2] * y[3] * n.sin(2*y[1]) - alph * y[0]

    a21 = 0
    a22 = m/3 + 3/2 * m1 * n.cos(y[1])**2
    b2  = 1/2 * ((m/3 + m1) * y[2]**2 + 3/2 * m1 * y[3]**2) * n.sin(2*y[1]) + 1/2 * (m*g/l * n.sin(y[1]) - c * n.sin(2*y[1]))

    dy[2] = (b1 * a22 - b2 * a12)/ (a11 * a22 - a12 * a21)
    dy[3] = (a11 * b2 - a21 * b1)/ (a11 * a22 - a12 * a21)
    
    return dy


J = 20
m = 2
m1 = 1
l = 1
r = 0.2
c = 20
alph = 30
g = 9.81

T = n.linspace(0, 10, 100)
y0 = [n.pi/6, n.pi/12, 0.1, 0]

Y = odeint(SystDiffEq, y0, T, (J,m,m1,l,r,alph,c,g))

Phi = Y[:,0]
Psi = Y[:,1] # n.sin(0.5*T) + 1.1
Phit = Y[:,2]
Psit = Y[:,3]

fgrp = p.figure()
plPhi = fgrp.add_subplot(4,1,1)
plPhi.plot(T, Phi)

plPsi = fgrp.add_subplot(4,1,2)
plPsi.plot(T, Psi)

Phitt = n.zeros_like(T)
Psitt = n.zeros_like(T)
for i in range(len(T)):
    Phitt[i] = SystDiffEq(Y[i], T[i], J,m,m1,l,r,alph,c,g)[2]
    Psitt[i] = SystDiffEq(Y[i], T[i], J,m,m1,l,r,alph,c,g)[3]

RA = (m + 3*m1)*l * (Psitt * n.cos(Psi) - Psit**2 * n.sin(Psi))/2

NK = -m/2 * l * (Psitt * n.sin(Psi) - Psit**2 * n.cos(Psi)) + (m+m1)*g

plRA = fgrp.add_subplot(4,1,3)
plRA.plot(T, RA)

plNK = fgrp.add_subplot(4,1,4)
plNK.plot(T, NK)

p.show()


#======= анимация системы =========
fgr = p.figure()
plt = fgr.add_subplot(1,1,1)
plt.axis('equal')

l = 1
r = 0.5
a = 3

h = 0.5
b = 2

plt.plot([0,0],[0,3])
plt.plot([0, a, a, 0],[h, h, h+b, h+b])

# Шаблон окружности
Alp = n.linspace(0, 2*n.pi, 100)
Xc = r * n.cos(Alp)
Yc = r * n.sin(Alp)

Xb = l * n.sin(Psi[0])
Yb = h+r

Disk = plt.plot(Xc + Xb, Yc + Yb)[0]

Xa = 0
Ya = h + r + l * n.cos(Psi[0])

AB = plt.plot([Xa, Xb],[Ya, Yb])[0]

# Шаблон пружины
# /\  /\  /\
#   \/  \/  \/
Np = 30
Xp = n.linspace(0,1,2*Np + 1)
Yp = 0.05 * n.sin(n.pi/2 * n.arange(2*Np + 1))

Pruzh = plt.plot(Xb + (a - Xb)*Xp, Yp + Yb)[0]

# Шаблон спиральной пружины
Ns = 3
r1 = 0.06
r2 = 0.1
numpnts = n.linspace(0,1,50*Ns + 1)
Betas = numpnts * (2*n.pi * Ns - Psi[0])
Xs = n.sin(Betas) * (r1 + (r2-r1)*numpnts)
Ys = n.cos(Betas) * (r1 + (r2-r1)*numpnts)

SpPruzh = plt.plot(Xs + Xb, Ys + Yb)[0]

def run(i):
    Xb = l * n.sin(Psi[i])
    Disk.set_data(Xc + Xb, Yc + Yb)
    Pruzh.set_data(Xb + (a - Xb)*Xp, Yp + Yb)
    Ya = h + r + l * n.cos(Psi[i])
    AB.set_data([Xa, Xb],[Ya, Yb])

    Betas = numpnts * (2*n.pi * Ns - Psi[i])
    Xs = n.sin(Betas) * (r1 + (r2-r1)*numpnts)
    Ys = n.cos(Betas) * (r1 + (r2-r1)*numpnts)
    SpPruzh.set_data(Xs + Xb, Ys + Yb)


    return

anim = FuncAnimation(fgr, run, frames = len(T), interval = 1)

p.show()







