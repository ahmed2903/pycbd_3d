
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from time import strftime

# Add the path to the directory containing 'pyxtools'
#sys.path.append(os.path.abspath('/Users/mohahmed/Documents/ConvergentBeam/Codes/'))
#sys.path.append(os.path.abspath('/home/mohahmed/Analysis_P11/'))

sys.path.append(os.path.abspath('/home/mohahmed/CBD_sim/CDIStudio/'))

from functions.crystal_fs import  cuboid_normals, ptycho_scan_volumes
from crystal import Crystal
from detector import Detector
from beam import Beam
from sequence import simulate_one_bragg_order

#Initialise Crystal
crys = Crystal()
crys.unit_cell_shape = (90,90,90)
crys.unit_cell_size = (4.025, 4.025, 4.025)
crys.crystal_size = (50,50,6)
crys.max_hkl = 1

atoms = {'Au': [0,0,0]}
cube_normals = cuboid_normals(crys.crystal_size)
padding = (5,5,5)

# Initialise Beam
beam = Beam()
beam.energy = 17
beam.lens_focal_length = 0
beam.focusing_lens_NA = 0.02
beam.set_convergent_kins(int(1e7))

#Optional
print("amplitudes .. ")

beam.amplitude_profile = {
   'gaussian_sigma' : 1,
   'absorption_coeff': 0,
   'tophat_radius': 1,
}
print("aberrations .. ")

beam.lens_aberrations = {
    'defocus': 0,
    'spherical':0,
    'coma':0,
    'astigmatism':0,
}
print("...done.")

beam.focus = beam.wavelength/beam.focusing_lens_NA * 5
beam_focus = (beam.focus // crys.unit_cell_size[:2]).astype(int)
stride = beam_focus + 1
ill_vol = ptycho_scan_volumes(crys.crystal_size, stride=stride, beam_focus=beam_focus, padding=padding)
print(f"beam_focus is {beam_focus}")
print(f"The illumination volumes are: {ill_vol}")

# Initialise Detector
det = Detector()
det.distance = 0.16
det.size = (0.12,0.12)
det.pixel_size = (75,75)


path = '/Users/mohahmed/Documents/ConvergentBeam/Codes/SimulationResults/'
time_str = strftime("%Y-%m-%d_%H.%M")
h5_file_name = f'{path}/simulation_{time_str}.h5'

#for vol in ill_vol:
    
simulate_one_bragg_order(
    crystal=crys,
    beam=beam, 
    detector=det,
    atoms = atoms,
    shape_normals = cube_normals,
    padding = padding, 
    
    ptycho_scan=True,
    illumination_vol = ill_vol[0],
    
    kin_kout_mapping = False,
    save_parameters = False,
    plot_2d_st = False,
    plot_3d_st=False,
    
    vector_map = False, 
    show = False, 
    validate_qs = False, 
    
    crystal_orientation = (45,0,18,0), # Chi, Mu, Eta, Phi
    #hkl = (-3,-1,2),
    hkl = (1,1,-1),
    one_hkl = True,
    
    mode = '3D', 
    grid_points = 80, 
    range_scale = .05,
    
    image_scale = .25,
    #low_pass_filter = 0.01,
    elasticity_tolerance = 1e-5,
    
    save_h5_file = h5_file_name
    #save_folder = '/Users/mohahmed/Documents/ConvergentBeam/Codes/SimulationResults'
)