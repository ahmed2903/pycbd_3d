
import numpy as np
from math import  pi



class Detector:
    
    def __init__(self):
        self._distance = None
        self._size = None
        self._pixel_size = None
        
        self.kouts = None
        
    @property
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self, distance):
        self._distance = distance
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size:tuple):
        
        self._size = size
        
    @property
    def pixel_size(self):
        
        return self._pixel_size
    
    @pixel_size.setter
    def pixel_size(self, pixel_size:tuple):
        
        self._pixel_size = pixel_size
        
    def set_kouts(self, wavelength):
        
        k_outs = crystal_to_detector_pixels_vector(self.distance, self.pixel_size, self.size, wavelength)

        self.kouts = k_outs
        

#################### Functions ###################

def crystal_to_detector_pixels_vector(detector_distance, pixel_size, detector_size, wavelength):
    
    """
    Assumes the optical axis is along the z-direction
    Left handed coordinate system
    
    Args: 
        detector_distance (float): distance from the centre of the crystal to the detector
        pixel_size (tuple): (x,y) size of the crystal (um)
        detector_size (tuple): (x,y) size of the crystal (m)
        wavelength (float): wavelength in Angstroms
        
    Returns:
        The outgoing wavevectors to every detector pixel
    """
    
    pixel_size = np.array(pixel_size) * 1e-6
    detector_distance = np.array(detector_distance)
    
    # number of pixels on the detector in each dimension
    nx,ny = np.floor(detector_size / pixel_size)

    x = np.arange(nx) - nx/2 + 0.5
    y = np.arange(ny) - ny/2 + 0.5
    
    x*= pixel_size[0]
    y*= pixel_size[1]
    
    xx,yy = np.meshgrid(x,y)
    
    zz = np.full(xx.shape, detector_distance)
    
    pixel_vectors = np.stack([xx, yy, zz], axis=2).reshape(-1,3) # this would be the real space vector
    
    unit_vectors = pixel_vectors/np.linalg.norm(pixel_vectors, axis = 1)[:,np.newaxis]
    
    k = 2.0*pi /wavelength
    
    k_out = k*unit_vectors
        
    return k_out