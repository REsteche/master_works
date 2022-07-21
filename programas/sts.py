# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:35:40 2022

@author: ruben.araujo
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

ststime = pd.read_csv('ststime_1.csv')
print('data labeled in data frame format')
#print(ststime.head(5))

xa = ststime[' eixo X']
ya = ststime[' eixo Y']

print(xa,ya)


dndt = pd.read_csv('dndt.csv')
xb = dndt[' eixo X']
yb = dndt[' eixo Y']

kijowski = pd.read_csv('J.csv')
xc = kijowski[' eixo X']
yc = kijowski[' eixo Y']

stsspace = pd.read_csv('stsspace.csv')
xd = stsspace[' eixo X']
yd = stsspace[' eixo Y']

golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = 10
fig_height = fig_width*golden_mean
fig_size = [fig_width,fig_height]

plt.rcParams.update({'font.family':'serif', 'font.size':'16', 'xtick.labelsize': '15',
                     'ytick.labelsize': '15' ,'figure.figsize': fig_size})

plt.plot(xa,ya,'-.',markersize=4, color='blue',label= 'sts_time(150,t)')
plt.plot(xd,yd,color='green',label= 'sts_space(150,t)')
plt.ylabel('$| \phi (t|x)|^2$')
plt.xlabel('$t$')
plt.legend()
plt.show()

plt.plot(xa,ya,'-.',markersize=4,label= 'sts_time(150,t)')
plt.plot(xc,yc,color='red',label= 'J(150,t)')
plt.ylabel('$| \phi (t|x)|^2$')
plt.xlabel('$t$')
plt.legend()
plt.show()

plt.plot(xa,ya,'.',markersize=4,color='blue',label= 'sts_time(150,t)')
plt.plot(xb,yb,color='orange',label= 'dN/dt')
plt.ylabel('$| \phi (t|x)|^2$')
plt.xlabel('$t$')
plt.legend()
plt.show()




