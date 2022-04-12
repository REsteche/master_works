# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 20:08:37 2022

@author: ruben
"""

from scipy.integrate import quad
from math import exp, cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt


#defina aqui as constantes: 
sigma = 0.5
hbar = 1
m = 1
P_0 = 0
i = 0 + 1j

#preparando a função que vai ser integrada para a integração, dividida em real e imaginária
#parte real: 
def re(p, t, x):
   RE = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*cos((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
   return RE
integral1 = np.vectorize(re)

#parte imaginária
def im(p,t,x): 
    IM = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*sin((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
    return IM
integral2 = np.vectorize(im)


#efetuando a integral em p e passando t para ser o argumento da função de onda 
def F(t,x):
    min = 0
    max = 15
    y, err = quad(integral1, min, max, args=(t,x))
    return y


def M(t,x):
    min = 0
    max = 15
    y, err = quad(integral2, min, max, args=(t,x))
    return y


def OP(t,x):
    return F(t,x) + M(t,x)*i 
OPvec = np.vectorize(OP)



#definindo as integrais 1 e 2 do meu \DeltaT
#integral de <T^2>
def R(t, x):
    return (t**2)*abs(OPvec(t,x))**2 
Rvec = np.vectorize(R)

#integral de <T> para fazer <T>^2
def L(t, x):
    return t*abs(OPvec(t,x))**2 
Lvec = np.vectorize(L)


#efetuando a integral em t e passando Xp para ser o argumento da função 1 e 2
def Q(x):
    min = -15
    max = 15
    y, err = quad(Rvec, min, max, args=(x))
    return y


def D(x):
    min = -15
    max = 15
    y, err = quad(Lvec, min, max, args=(x))
    return y

#calculando agora DeltaT como (<T^2> - <T>^2)^-(1/2)
def Sol(x):
    return sqrt(Q(x) - D(x)**2) 
Sol = np.vectorize(Sol)


#organizando o plot da função em questão 
X = np.linspace(-10, 10, 200)
plt.plot(X, Sol(X))
plt.title('Width of the free wave packet as a function of position')
plt.ylabel('$\Delta T$')
plt.xlabel('$x$')
plt.show()