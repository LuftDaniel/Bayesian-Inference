# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy import random
from scipy import special as special
from scipy import integrate as integrate
import matplotlib.patches as mpatches
import time

p = 0.6
#Wahrscheinlichkeit auf Kopf
a = 0.5
b = 0.5
#Parameter des Priors: 1,1 Bayes, 0.5,0.5 Jeffrey, 0.00001,0.00001 Haldane, oder 2,2
n = 32
#Größe des Samples


def Wurf(m):
     # n ist Anzahl der Würfe
     
     return np.random.binomial(1,p,m)


sample = Wurf(n)

 
def beta_density(x):
     # a priori Verteilungsdichte mit Parametern a, b
     
     return special.gamma(a+b)/(special.gamma(a)*special.gamma(b)) * x**(a-1) * (1-x)**(b-1)


def posterior(x, data):
     
     heads = sum(data == 1)
     tails = sum(data == 0)
     
     return x**(heads + a - 1) * (1-x)** (tails + b -1 ) * special.gamma(a + heads + tails +b)/(special.gamma(a + heads)*special.gamma(b + tails))



for j in range(0,n+1,4):
     #Posteriorupdate
     plt.figure(j)
     plt.plot(np.linspace(0,1,100), beta_density(np.linspace(0,1,100)))
     plt.plot(np.linspace(0,1,100), posterior(np.linspace(0,1,100), sample[0:j]))
     
     heads_count = sum(sample[0:j] == 1)
     heads_string = str(heads_count)
     tails_count = sum(sample[0:j] == 0)
     tails_string = str(tails_count)
     
     patch1 = mpatches.Patch( color = 'white', label= "Heads: " + heads_string)
     patch2 = mpatches.Patch( color = 'white', label= "Tails: " + tails_string)
     plt.legend(handles = [patch1, patch2])
     
     plt.show()
     plt.close()
     



