import matplotlib.pyplot as p
from matplotlib.animation import FuncAnimation
import numpy as n

T = n.linspace(0, 10, 1000)

Psi = n.sin(0.5*T) + 1.1

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

fgr.show()








