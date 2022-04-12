# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:40:08 2021

@author: ruben
"""

import quadpy
import math as math
from math import e
from math import pi
from math import sqrt
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import cmath

#defina aqui as constantes: 
sigma = 1
hbar = 1
m = 1
P_0 = 0
Xp = 20
i = 0 + 1j


#preparando a função que vai ser integrada para a integração 
def f(x, t):
   res = (sigma**2)/(2*pi*hbar)**(3/2)*x*e**(-(P_0 + hbar*(x**2)/(2*m))*(sigma**2)*(P_0 + hbar*(x**2)/(2*m))/(2*hbar**2))*cmath.exp((i/hbar)*x*Xp - i*(x**2)*t/(2*m*hbar))
   return res
integral = np.vectorize(f)

#efetuando a integral 
def F(t):
    xmin = 0
    xmax = 10
    y,err = quadpy.quad(lambda x: integral(x,t), xmin, xmax)
    return y
Fvec = np.vectorize(F)

t = np.linspace(0,40,200)
plt.scatter(t, abs(Fvec(t))**2, s=2)
plt.title('Numerical evolution of the probability density of Gaussian wave packet')
plt.ylabel('$|\phi(t|x)|^2$')
plt.xlabel('$t$')




