# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:25:13 2022

@author: ruben
"""

from scipy.integrate import quad
from math import exp, cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

#defina aqui as constantes: 
sigma = 1
hbar = 1
m = 1
P_0 = 1
V0 = 1
i = 0 + 1j
n = 300
irio = 400
piu = np.zeros(irio)
valor_x = np.zeros(n)
maximum = np.zeros(n)
tuntime = np.zeros(n)


#executando o algoritmo que calcular \psi e tira seus maximos 
for xxx in range (n):
    #preparando a função que vai ser integrada para a integração, dividida em real e imaginária
    #parte real:
    x = xxx*0.10
    def re(p, t):
       #RE = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*cos((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
       RE = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*cos((1/hbar)*sqrt(((p**2) - V0))*x - 1*(p**2)*t/(2*m*hbar))
       return RE
    integral1 = np.vectorize(re)

    #parte imaginária
    def im(p,t): 
        #IM = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*sin((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
        IM = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*sin((1/hbar)*sqrt(((p**2) - V0))*x - 1*(p**2)*t/(2*m*hbar))
        return IM
    integral2 = np.vectorize(im)


    #efetuando a integral em p e passando t para ser o argumento da função de onda 
    def reint(t):
        min = 1
        max = 40
        y, err = quad(integral1, min, max, args=(t))
        return y


    def imint(t):
        min = 1
        max = 40
        y, err = quad(integral2, min, max, args=(t))
        return y

    #calculando a função densidade de probabilidade
    def OP(t):
        return abs(reint(t) + imint(t)*i)**2 
    OPvec = np.vectorize(OP)
    
    
    #integrando em t para obter a função de normalização
    def norm(t):
        min = -40
        max = 40
        y, err = quad(OPvec, min, max)
        return y


    #normalizando
    def graph(t):
        return OP(t)/norm(t)
        
    
    for k in range (irio):
        piu[k] = graph(k*0.10) 
    
    
    tuntime[xxx] = np.argmax(piu)
    maximum[xxx] = np.amax(piu)
    valor_x[xxx] = xxx    
            


#preparando aqui os plots da função, e da sua integral 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

axes[0].set_title('Maximum of $|\phi (t)|^2$')
axes[0].set_ylabel('$MAX(|\phi(t)|^2$)') 
axes[0].set_xlabel('$x$')                         
axes[0].plot(valor_x/10,maximum)

axes[1].set_title('Tunneling time')
axes[1].set_ylabel('time that maximizes $|\phi (t)|^2$') 
axes[1].set_xlabel('$x$')
axes[1].plot(valor_x/10,tuntime)

fig.tight_layout()


