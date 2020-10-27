## Some convenient helper functions

import numpy as np
from numba import njit

def phi_to_L_simplebi_2D(phi=0.85,N=16000):
    return np.power(0.5*N*np.pi*((5/12)**2 + (7/12)**2)/phi,1/2) 

def phi_to_L_simplebi_3D(phi=0.65,N=16000):
    return np.power((2/3)*N*np.pi*((5/12)**3 + (7/12)**3)/phi,1/3)

def phi_to_L_simplemono_2D(phi=0.85,N=16000):
    return np.power(.5*N*np.pi/phi,1/3)

def phi_to_L_simplemono_3D(phi=0.65,N=16000):
    return np.power((2/3)*N*np.pi/phi,1/3)

@njit
def rotate(pos, alpha):
    mat = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]], dtype=np.float32)
    new_pos = np.zeros_like(pos)
    for i in np.arange(len(pos)):
        p = pos[i]
        new_p = np.dot(mat, p)
        new_pos[i] = new_p
    return new_pos