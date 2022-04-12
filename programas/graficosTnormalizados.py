# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:56:08 2022

@author: ruben
"""

from scipy.integrate import quad
from math import exp, cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import cmath

#defina aqui as constantes: 
i = 0 + 1j


#função de plot 
def plot_results(sigma = 1, x = 1, m = 1, P_0 = 1, hbar = 1, V0 = 1):

    #Para o gráfico da barreira de potencial: 
    #energia menor que potencial: P_0 = 1 
    #energia maior que potencial: P_0 = 2
    #energia igual ao potencial: P_0 = sqrt(2)
    #preparando a função que vai ser integrada para a integração, dividida em real e imaginária
    #parte real: 
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
        max = 20
        y, err = quad(integral1, min, max, args=(t))
        return y


    def imint(t):
        min = 1
        max = 20
        y, err = quad(integral2, min, max, args=(t))
        return y

    #calculando a função densidade de probabilidade
    def OP(t):
        return abs(reint(t) + imint(t)*i)**2 
    OPvec = np.vectorize(OP)

    #integrando em t para obter a função de normalização
    def norm(t):
        min = -20
        max = 20
        y, err = quad(OPvec, min, max)
        return y


    #normalizando
    def graph(t):
        return OP(t)/norm(t)
    Gvec = np.vectorize(graph)
    
    return Gvec 
     
X0 = plot_results(1,0,1,1,1) 
X1 = plot_results(1,2,1,1,1) 
X2 = plot_results(1,4,1,1,1) 
X3 = plot_results(1,6,1,1,1)    
    

#preparando o plot
t = np.linspace(-5,20,200)
plt.plot(t, X0(t), label='x=0')
plt.plot(t, X1(t), label='x=3')
plt.plot(t, X2(t), label='x=6')
plt.plot(t, X3(t), label='x=9')
plt.legend()
plt.title('Numerical evolution of the probability density of Gaussian wave packet $V_0 > \epsilon$')
plt.ylabel('$|\phi(t|x)|^2$')
plt.xlabel('$t$')













