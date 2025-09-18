import numpy as np
from aberrations import *
from math import pi 
import utils as ut

class Beam:
    
    def __init__(self):
        
        self._wavelength = None # wavelength in Angs
        self._energy = None
        self._flux = None
        self.kins = None
        self._lens_fl = None
        self._lens_NA = None
        self.num_vectors  = None
        self._lens_aberrations = None
        self._amplitude_profile = None
        self.amplitude_coefficients = None
        self.aberration_coefficients = None
    
    @property
    def energy(self):
        """
        Energy of the beam in keV
        """
        return self._energy
    
    @energy.setter
    def energy(self, energy):
        """
        Energy of the beam in keV
        """
        self._energy = energy
        self._wavelength = energy2wavelength_a(energy)
        
    @property
    def wavelength(self):
        """
        Wavelength of the beam in keV
        """
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, wavelength):
        """
        Wavelength of the beam in keV
        """
        self._wavelength = wavelength
        
    @property
    def flux(self):
        """
        Flux in photons/um^2/s
        """
        return self._flux

    @flux.setter
    def flux(self, flux):
        """
        Sets flux in photos/m^2/s and converts it to photons/um^2/s
        """
        self._flux = float(self.flux*(10.0**(-12)))
    
    
    @property
    def focusing_lens_NA(self):
        """
        numerical aperture of the lens
        """
        return self._lens_NA
    
    @focusing_lens_NA.setter
    def focusing_lens_NA(self, NA):
        """
        Set the numerical aperture of the lens
        """
        self._lens_NA = NA
        
    @property
    def lens_focal_length(self):
        """
        numerical aperture of the lens
        """
        return self._lens_fl
    
    @lens_focal_length.setter
    def lens_focal_length(self, focal_length):
        """
        Set the numerical aperture of the lens
        """
        self._lens_fl = focal_length
    
    @property
    def lens_aberrations(self):
        """
        Lens aberrations
        """
        return self._lens_aberrations
    
    @lens_aberrations.setter
    def lens_aberrations(self, coefficients):
        """
        Set the aberrations coefficients of the lens 
        Args:
            coefficients (dictionary): dictionary of the aberration coefficient values from the options
            
                Options:
                    defocus
                    spherical
                    coma
                    astigmatism
        """
        
        phase_error = combined_aberrations(self.kins[:,0], self.kins[:,1], coefficients)
        
        self._lens_aberrations = phase_error        
        self.aberration_coefficients = coefficients
        
    @property
    def amplitude_profile(self):
        """ beam amplitude profile """
        return self._amplitude_profile
    
    @amplitude_profile.setter
    def amplitude_profile(self, coefficients):
        """
        Set the ampltiude coefficients of the incident beam 
            
        Args:
            coefficients (dictionary): dictionary of the aberration coefficient values from the options
                Options:
                    gaussian_sigma 
                    absorption_coeff
                    tophat_radius
            
        """
        
        sigma = coefficients['gaussian_sigma']
        absorption_coeff = coefficients['absorption_coeff']
        radius = coefficients['tophat_radius']
        
        profiles = [
                    lambda kx, ky: gaussian_amplitude(kx, ky, sigma=sigma),
                    lambda kx, ky: top_hat_amplitude(kx, ky, radius = radius),
                    lambda kx, ky: absorption_amplitude(kx, ky, absorption_coeff=absorption_coeff)
                ]
        
        amp = combined_amplitude(self.kins[:,0], self.kins[:,1], profiles)
        
        self._amplitude_profile = amp
        self.amplitude_coefficients = coefficients
    
    def set_convergent_kins(self, num_vectors=100):
        """
        Generate an array of incoming k-vectors (incident wave vectors)
        
        Parameters:
        - NA: Numerical aperture (NA)
        - focal_length: Focal length of the lens (mm)
        - num_vectors: Number of k-vectors to generate (default = 100)
        
        Returns:
        - Array of incoming k-vectors (shape: [num_vectors, 3])
        """
        
        k_vectors = convergent_kins(self.wavelength, self.focusing_lens_NA, self.lens_focal_length, num_vectors)
        
        self.num_vectors = num_vectors
        self.kins = k_vectors
        
        self._amplitude_profile = uniform_amplitude(k_vectors[:,0], k_vectors[:,1])
        self._lens_aberrations = np.zeros_like(k_vectors[:,0])
        
        

########################### Functions ###########################

def compute_kout_from_G_kin(G_arr, kin_arr):
    """
    A functiont that computes k_out from the reciprocal lattice vectors, and k_in
    
    k_out = k_in + G

    Args:
        G_arr (np.ndarray): The reciprocal lattice vectors
        kin_arr (np.ndarray): The incoming wave vectors
    """
    
    
    k_out = G_arr[:, None, :] + kin_arr[None, :, :]
    
    k_out = k_out.reshape(-1,3)
    
    # Generate the indices for the kin vectors
    kin_indices = np.tile(np.arange(len(kin_arr)), len(G_arr))
    Garr_indices = np.repeat(np.arange(len(G_arr)), len(kin_arr))
    
    return k_out, kin_indices, Garr_indices
    
def energy2wavelength_a(energy_kev: float) -> float:
    """
    Converts energy in keV to wavelength in A
    wavelength_a = energy2wave(energy_kev)
    lambda [A] = h*c/E = 12.3984 / E [keV]
    """
    
    # Electron Volts:
    E = 1000 * energy_kev * ut.echarge

    # SI: E = hc/lambda
    lam = ut.hplanck * ut.c / E # in meters
    wavelength_a = lam / ut.Ang # in angstroms

    return wavelength_a

def convergent_kins(wavelength, NA, focal_length, num_vectors=100):
    """
    Generate an array of incoming k-vectors (incident wave vectors)

    Parameters:
    - NA: Numerical aperture (NA)
    - focal_length: Focal length of the lens (mm)
    - num_vectors: Number of k-vectors to generate (default = 100)

    Returns:
    - Array of incoming k-vectors (shape: [num_vectors, 3])
    """
    # Calculate the maximum scattering angle from the numerical aperture
    theta_max = np.arcsin(NA)

    # Generate random directions within the cone defined by the NA
    phi = np.random.uniform(0, 2 * np.pi, num_vectors)  # Random azimuthal angle (0 to 2*pi)
    
    u = np.random.uniform(0,1, num_vectors)
    theta = np.arccos(1-u*(1-np.cos(theta_max)))
    
    #theta = np.random.uniform(0, theta_max, num_vectors)  # Random polar angle (0 to theta_max)

    # Convert spherical coordinates to Cartesian coordinates for the k-vectors
    k_vectors = np.zeros((num_vectors, 3))
    k_vectors[:, 0] = np.sin(theta) * np.cos(phi)  # x component
    k_vectors[:, 1] = np.sin(theta) * np.sin(phi)  # y component
    k_vectors[:, 2] = np.cos(theta)  # z component

    # Normalize to have unit length (magnitude of k-vector should be 2*pi / wavelength)
    k_vectors /= np.linalg.norm(k_vectors, axis = 1, keepdims=True)
    k_magnitude = 2.0*pi / wavelength
    k_vectors *= k_magnitude

    return k_vectors
    
