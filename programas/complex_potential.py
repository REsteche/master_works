# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:00:31 2022

@author: ruben
"""

from math import exp, cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import quadpy

#definindo unidade imaginária:
i = 0 + 1j

#definindo constantes 
p_0 = 1
delta = 5 
x_0 = -50 
m = 0.5
L = 1 
hbar = 1
k_0 = 1


#função vetorizada para auxiliar na visualização da condição inicial 

def psi_init(x) :
    return 1/(2*pi*(delta)**2)**(0.25)*np.exp(-(x-x_0)**2/(4*delta**2))*np.exp(i*p_0*x/hbar)

vecpsi_i = np.vectorize(psi_init)

#assumindo que vamos pegar agora posicão em x = 10, precisamos tomar a transformada de fourier da
#condição incial e então integrar essa nova função com o propagador para gerar psi(x,t)  
#apliquei de forma bem sucedida a transfromada de fourier abaixo e obtive Phi(x,0) = phi(k) 


N = 150
x = np.arange(-75, 75, 10./(2*N))

cond_init = 1/(2*pi*(delta)**2)**(0.25)*np.exp(-(x-x_0)**2/(4*delta**2))*np.exp(i*p_0*x/hbar)

def phi(k) :
    return abs(fft(cond_init)) / np.sqrt(len (cond_init)) 
               
k = np.arange(-75, 75, 10./(2*N))
tempo = 40  

evolve = phi(k)*np.exp(i*hbar*tempo*(k**2)/(2*m))
psi_evolve = abs(fft(evolve)) / np.sqrt(len (evolve))

#organziando o plot de phi para verificar se tudo funcionou como esperado:  
plt.plot(x, abs(vecpsi_i(x))**2, label='initial condition $|\psi(x,0)|^2$')
#plt.plot(x, phi_init, label='fourier transform $\phi(k,0)$')
plt.plot(x, abs(psi_evolve)**2, label='$|\psi(x,t)|^2$ for $t>0$')
plt.legend() 
plt.grid()
plt.show()

#lembrar que o plot final é a integral da densidade de probabilidade do psi obtido acima!!>>(dN/dt)
#para isso, manter em mente que <t>_n = int -dNdt * t dt / int -dNdt dt, onde 
# dNdt =  <V1>/hbar = 1/(hbar*L) * int V1(x)*|psi(x,t)|^2 dx
#V1(x) = 2/((x - 1)*(x + 1/(1 + i*k_0))) -> imaginary part of potential inside potential 


prob_dens = abs(psi_evolve)**2 

#transformando o array em uma função de seu próprio indice 
def funpd(x):
    tam = x.astype(int)
    return prob_dens[tam]
funpd_vec = np.vectorize(funpd)


#definindo o potencial complexo para um polinômio quadrado que fite psi_2
def V1(x) :
    return 2/((x - 1)*(x + 1/(1 + i*k_0)) + 1e-10) 
v1_vec = np.vectorize(V1)
 

#juntando potencial e dens pro valor esperado
def dNdt(x):
    return funpd(x)*V1(x) 
  
dndt_vec = np.vectorize(dNdt)

#fazendo a integral em x pra ter uma função em t
def plot(t):
    min = 0
    max = 1
    y,err = quadpy.quad(dNdt, min, max)
    return -y
plot_vec = np.vectorize(plot) 


T = np.arange(0, 1, 10./(2*N))
plt.plot(T, abs(plot_vec(T))**2)
plt.show()




    

