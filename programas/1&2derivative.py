# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:46:11 2022

@author: ruben
"""

from scipy.integrate import quad
from math import exp, cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt

#defina aqui as constantes: 
sigma = 1
hbar = 1
m = 1
P_0 = 0
i = 0 + 1j
n = 20
x = 10
piu = np.zeros(100)

#definindo uma derivada para o algortimo de máximos
def derivative(f,a,method='central',h=0.01):
    
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
        
       
        
#preparando a função que vai ser integrada para a integração, dividida em real e imaginária
#parte real: 
def re(p, t):
    RE = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*cos((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
    return RE
integral1 = np.vectorize(re)

#parte imaginária
def im(p, t): 
    IM = (sigma**2)/(2*pi*hbar)**(3/2)*p*exp(-(P_0 + hbar*(p**2)/(2*m))*(sigma**2)*(P_0 + hbar*(p**2)/(2*m))/(2*hbar**2))*sin((1/hbar)*p*x - 1*(p**2)*t/(2*m*hbar))
    return IM
integral2 = np.vectorize(im)


#efetuando a integral em p e passando t para ser o argumento da função de onda 
def F(t):
    min = 0
    max = 10
    y, err = quad(integral1, min, max, args=(t))
    return y


def M(t):
    min = 0
    max = 10
    y, err = quad(integral2, min, max, args=(t))
    return y

def OP(t):
    return abs(F(t) + M(t)*i)**2 
OPvec = np.vectorize(OP)
    
def der1(t):
    deu = derivative(OP, t, method='central',h=0.01)
    return deu
der1vec = np.vectorize(der1)

def der2(t):
    deu = derivative(der1, t, method='central',h=0.01)
    return deu
der2vec = np.vectorize(der2)

for k in range (100):
    piu[k] = OP(k) 
        
maximum = np.amax(piu)  
print(maximum) 

#organizando o plot da função em questão 
t = np.linspace(-10,20,200)
plt.scatter(t,OPvec(t), s=2, label='function')
plt.plot(t,der1vec(t),label='1st derivative')
plt.plot(t,der2vec(t),label='2nd derivative')
plt.legend()
plt.title('function and derivative')