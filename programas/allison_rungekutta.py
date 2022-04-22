# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:17:55 2021

@author: ruben
"""

import numpy as np
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, system, n_steps, n_eqs, h):
        #Configurações iniciais
        self.n_steps = n_steps
        self.n_eqs = n_eqs
        self.n_rep = self.n_steps - 1
        self.h = h #step size (time)
        self.system = system
        
    #Definição do sistema de equações:
    def _F(self, t, n, n_0, i):
        n_ = np.copy(n)
        for j in range(self.n_eqs):
            n_[j][i] = n[j][i] + n_0[j]
        f = np.zeros(self.n_eqs)
        
        f[0] = self.system.R_prime_pump*(1 - n_[0][i]) - (self.system.tau_p/self.system.tau_21)*n_[0][i]*(1 + 2*n_[1][i])

        f[1] = - n_[1][i] + (n_[0][i]/self.system.n_th)*(n_[1][i] + self.system.eta_SE)

        return f
    
    
    #Constantes definidas no método de Runge-Kutta
    
    def _update_n(self, n,k1,k2,k3,k4,i):
        for j in range(self.n_eqs):
            n[j][i+1] = n[j][i] + (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])/6
            
    def _calculate(self, n, t):
        #matriz-solução; tempos discretos
        for i in range(self.n_rep):
            k1= self._F(t[i]            , n , np.zeros(self.n_eqs)  , i)*self.h
            k2= self._F(t[i]+self.h/2   , n , k1/2                  , i)*self.h
            k3= self._F(t[i]+self.h/2   , n , k2/2                  , i)*self.h
            k4= self._F(t[i]+self.h     , n , k3                    , i)*self.h
            
            self._update_n(n,k1,k2,k3,k4,i)
            t[i+1] = t[i] + self.h
        
        return t,n

    def run(self, init_cond):
        n = np.zeros((self.n_eqs,self.n_steps))
        t = np.zeros(self.n_steps)
        #Condições iniciais das variáveis
        n[0][0] = init_cond[0]
        n[1][0] = init_cond[1]
        
        return self._calculate(n, t)
    
class System():
    def __init__(self, system):
        if system == 'Nd:YAG':
            #Constantes prévias
            R1 = 1
            R2 = 0.95
            l_g = 1 #m
            sigma_0 = 5.92e-23 #m²
            N = 1e25 #part/m³
            tau = 3e-6 #s
            R_pump = 1.8 #s-1
            #Definições do sistema
            self.tau_p = tau/(1-R1*R2)
            self.R_prime_pump = R_pump*self.tau_p
            self.tau_21 = 1.2e-3
            self.eta_SE = 1e-12
            self.n_th = (1-R1*R2)/(2*l_g*sigma_0*N)
            
            
# SISTEMA E PARÂMETROS DA SIMULAÇÃO
NdYAG = System('Nd:YAG')
Sim = Simulation(NdYAG,1200,2,15)

def plotPop(t, n, labels=None, normalize=False):
    if normalize == True:
        for i in range(len(n)):
            n[i] = (np.asarray(n[i])/max(n[i])).tolist()
    fig, ax = plt.subplots(2,1)
    if labels==None: labels=range(len(n))
    for i in range(len(n)):
        ax[i].plot(t, np.asarray(n[i]),'-',label=labels[i])
        ax[i].set_xlabel(r't$^\prime$ ($t/\tau_p$)', size='x-large')
        ax[i].tick_params(direction='in',which='both')
        
    ax[0].set_ylabel('$N_2/N$', size='x-large')
    ax[1].set_ylabel('$I/I_{sat}$', size='x-large')
    
def runSimulation():
    t,n = Sim.run([0,0])
    plotPop(t,n, normalize=False)
    
    
runSimulation()