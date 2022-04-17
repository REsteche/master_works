# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:00:22 2022

@author: ruben
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import quadpy


#definindo constantes 
p_0 = 1
delta = 5 
x_0 = -50 
m = 0.5
L = 1 
hbar = 1
k_0 = 1


def psi(x,t):
    #função de onda analiticamente evoluída da condição inicial do muuga 
    return (8*delta**2/pi)**(1/4)*np.exp(1j*(k_0*(x-x_0) - hbar*k_0**2*t/(2*m)\
    -np.arctan(2*hbar*t/(4*m*delta**2))/2))/(16*delta**4 + 4*hbar**2*t**2/m**2)**(1/4)\
    *np.exp(-((x-x_0) - hbar*k_0*t/(m))**2/(4*delta**2 + 2*1j*hbar*t/m)) 
        

def dens_prob(x,t):
    #densidade de probabilidade referente a função de onda anteriormente definida
    return (2/(pi*4*delta**2))**(1/2)*(1/(1 + 4*hbar**2*t**2/(m**2*16*delta**4))**(1/2))\
    *np.exp(-8*delta**2*((x-x_0) - hbar*k_0*t/(m))**2/(16*delta**4 + 4*hbar**2*t**2/m**2))  

#lembrar que o plot final é a integral da densidade de probabilidade do psi obtido acima!!>>(dN/dt)
#para isso, manter em mente que <t>_n = int -dNdt * t dt / int -dNdt dt, onde 
# dNdt =  <V1>/hbar = 1/(hbar*L) * int V1(x)*|psi(x,t)|^2 dx
#V1(x) = 2/((x - 1)*(x + 1/(1 + i*k_0))) -> imaginary part of potential inside potential 

def V1(x):
    return 2/(hbar*(x - 1)*(x + 1/(1 + 1j*k_0))) 

def integrand(x,t):
    return V1(x)*dens_prob(x,t)
inte_vec = np.vectorize(integrand)

def dNdt(t):
    lmin = 0.001
    lmax = 0.999
    y, err = quadpy.quad(lambda x: inte_vec(x,t), lmin, lmax)
    return -y
dndt_vec = np.vectorize(dNdt) 

def ajust(t):
    return dNdt(t)*t
ajust_vec = np.vectorize(ajust)

def num(t):
    lmin = -20
    lmax = 20
    y, err = quadpy.quad(ajust_vec, lmin, lmax)
    return y
  
def den(t):
    lmin = -20
    lmax = 20
    y, err = quadpy.quad(dndt_vec, lmin, lmax)
    return y

def dndt_norm(t): 
    return num(t)/den(t) 
plot_vec = np.vectorize(dndt_norm)
    
T = np.arange(10, 40, 0.001)
plt.plot(T,dndt_vec(T), label = 'unnormalized')
plt.plot(T,plot_vec(T), label = 'normalized')
plt.legend()
plt.grid()
plt.ylabel('absortion rate -dN/dt')
plt.xlabel('time') 
plt.show()








