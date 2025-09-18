import numpy as np 
from math import cos, sin, pi, fabs
import time 
import functools

hplanck = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
Ang = 1e-10  # m Angstrom
echarge = 1.6021733E-19  # C  electron charge
emass = 9.109e-31  # kg Electron rest mass
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)
Cu = 8.048  # Cu-Ka emission energy, keV
Mo = 17.4808  # Mo-Ka emission energy, keV

def time_it(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Compute execution time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result  # Return original function result
    return wrapper
    

def indexQ(Qvec, rlv):
    """_summary_

    Args:
        Qvec (vector): Q vector (n,3)
        rlv (vector): Reciprocal lattice vectors (3,3) (astar, bstar, cstar)
    """

    HKL = np.round(np.dot(Qvec, np.linalg.inv(rlv))).astype(int)
    return HKL

def X_Rot(x, right_handed = True):
    """
    Right handed mu rotation about x axis
    """
    x = np.deg2rad(x)
    if right_handed == False:
            x = -x 

    return np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)] ]) 

def Y_Rot(y, right_handed = True):

    
    if right_handed == False:
        y = -y
    
    y = np.deg2rad(y)

    return np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ])

def Z_Rot(z, right_handed = True): 

    z = np.deg2rad(z)
    if right_handed == False:
        z = -z

    return np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ])

    
def StanRotMat(chi, mu, eta, phi):        
    
    #Standard System
    mu_rot = X_Rot(mu, right_handed = True)
    eta_rot = Z_Rot(eta, right_handed = False)
    chi_rot = Y_Rot(chi, right_handed = True)
    phi_rot = Z_Rot(phi, right_handed = False)
    
    rotmat = np.dot(mu_rot, np.dot(eta_rot, np.dot(chi_rot, phi_rot)))

    return rotmat


