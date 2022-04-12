# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:27:02 2022

@author: ruben
"""

import quadpy
from math import e
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import cmath

#definindo unidade imaginária:
i = 0 + 1j

#Para o gráfico da barreira de potencial: 
#energia menor que potencial: P_0 = 1 
#energia maior que potencial: P_0 = 2
#energia igual ao potencial: P_0 = sqrt(2)
#criando a função para o plot
def plot_results(sigma = 1, Xp = 1, m=1, P_0 = 1, hbar = 1, V0 = 1):
    def f(x, t):
        #res = (sigma**2)/(2*pi*hbar)**(3/2)*x*e**(-(P_0 + hbar*(x**2)/(2*m))*(sigma**2)*(P_0 + hbar*(x**2)/(2*m))/(2*hbar**2))*cmath.exp((i/hbar)*x*Xp - i*(x**2)*t/(2*m*hbar))
        #res = (sigma**2)/(2*pi*hbar)**(3/2)*x*e**(-(P_0 + hbar*(x**2)/(2*m))*(sigma**2)*(P_0 + hbar*(x**2)/(2*m))/(2*hbar**2))*cmath.exp((i/hbar)*sqrt(2*m)/(2*k)*(Xp*k*cmath.sqrt((Xp**2)/(2*m) -(Xp**2)*k)+ (x**2)/(2*m)*np.arcsin(Xp*cmath.sqrt(k*2*m/(x**2)))*sqrt(k)) - i*(x**2)*t/(2*m*hbar))
        res = (sigma**2)/(2*pi*hbar)**(3/2)*x*e**(-(P_0 + hbar*(x**2)/(2*m))*(sigma**2)*(P_0 + hbar*(x**2)/(2*m))/(2*hbar**2))*cmath.exp((i/hbar)*cmath.sqrt(2*m*((x**2)/(2*m) - V0))*Xp - i*(x**2)*t/(2*m*hbar))
        return res
    integral = np.vectorize(f)

    #efetuando a integral 
    def F(t):
        xmin = 0
        xmax = 15
        y,err = quadpy.quad(lambda x: integral(x,t), xmin, xmax)
        return y
    
    def OP(t):
        return abs(F(t))**2 
    OPvec = np.vectorize(OP)
    
    def norm(t):
        min = -15
        max = 15
        y, err = quadpy.quad(OPvec, min, max)
        return y
    
    #normalizando
    def graph(t):
        return OP(t)/norm(t)
    Gvec = np.vectorize(graph)
    
    return Gvec 

X0 = plot_results(1,0,1,1,1,1) 
X1 = plot_results(1,3,1,1,1,1) 
X2 = plot_results(1,6,1,1,1,1) 
X3 = plot_results(1,9,1,1,1,1)
#X4 = plot_results(1,12,1,1,1)
#X5 = plot_results(1,15,1,1,1) 

t = np.linspace(-5,20,200)
plt.plot(t, X0(t), label='x=0')
plt.plot(t, X1(t), label='x=3')
plt.plot(t, X2(t), label='x=6')
plt.plot(t, X3(t), label='x=9')
#plt.plot(t, abs(X4(t))**2, label='x=12')
#plt.plot(t, abs(X5(t))**2, label='x=15')
plt.legend()
plt.title('Normalized numerical evolution of the probability density of Gaussian wave packet $V_0 > \epsilon$')
plt.ylabel('$|\phi(t|x)|^2$')
plt.xlabel('$t$') 