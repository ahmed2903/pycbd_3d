import numpy as np

#################################### Amplitude and Aberration Profiles for k_ins ####################################

####### probe

def probe_envelope(R_i, position=(0, 0), width=5):
    """
    R_i: (N, 3) unit cell positions
    center: (x, y) focal center in real space
    width: determines beam size (in A)

    Returns: amplitude modulation at each Ri
    """
    x, y = R_i[:, 0], R_i[:, 1]
    dx = x - position[0]
    dy = y - position[1]
    r = np.sqrt(dx**2 + dy**2)

    # Approximate sinc-shaped beam
    beam = (np.sinc(r / width))**2  

    return beam


###### pupil 
def apply_aberattions_to_kins(kins, amplitude_profile=None, phase_aberration=None):
    """
    Generate incoming k-vectors for a convergent beam, including pupil function effects.

    Parameters:
    - kins: the kin vector 
    - amplitude_profile: Function describing the amplitude distribution across the pupil.
                         Takes (kx_norm, ky_norm) as input and returns amplitude.
    - phase_aberration: Function describing the phase aberrations across the pupil.
                        Takes (kx_norm, ky_norm) as input and returns phase (in radians).

    Returns:
    - k_vectors: Array of shape (num_vectors, 3) with each row as a k-vector.
    - weights: Array of shape (num_vectors,) with amplitude and phase weights for each k-vector.
    """
    
    kx = kins[:,0]
    ky = kins[:,1]

    # Initialize weights (amplitude and phase)
    weights = np.ones(kins.shape[0], dtype=complex)

    # Apply amplitude profile if provided
    if amplitude_profile is not None:
        weights *= amplitude_profile(kx, ky)

    # Apply phase aberrations if provided
    if phase_aberration is not None:
        phase = phase_aberration(kx, ky)
        weights *= np.exp(1j * phase)  # Multiply by complex phase factor

    # Combine k-vectors and weights
    return kins, weights
    
# Phase Aberrations:
def defocus_aberration(kx, ky, defocus_coeff):
    """
    Defocus aberration: quadratic phase error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - defocus_coeff: Coefficient controlling the strength of defocus.
    
    Returns:
    - Phase error (in radians).
    """
    return defocus_coeff * (kx**2 + ky**2)

def spherical_aberration(kx, ky, spherical_coeff):
    """
    Spherical aberration: quartic phase error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - spherical_coeff: Coefficient controlling the strength of spherical aberration.
    
    Returns:
    - Phase error (in radians).
    """
    return spherical_coeff * (kx**2 + ky**2)**2

def coma_aberration(kx, ky, coma_coeff):
    """
    Coma aberration: linear in one direction, quadratic in the other.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - coma_coeff: Coefficient controlling the strength of coma.
    
    Returns:
    - Phase error (in radians).
    """
    return coma_coeff * kx * (kx**2 + ky**2)

def astigmatism_aberration(kx, ky, astigmatism_coeff):
    """
    Astigmatism aberration: quadratic phase error with asymmetry.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - astigmatism_coeff: Coefficient controlling the strength of astigmatism.
    
    Returns:
    - Phase error (in radians).
    """
    return astigmatism_coeff * ( kx * ky )

def random_error_profile(kx, ky, amplitude=0.1):
    """
    Random layer placement error.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - amplitude: Amplitude of the random error.
    
    Returns:
    - Random phase error (in radians).
    """
    return amplitude * np.random.normal(size=kx.shape)

def combined_aberrations(kx, ky, coefficients):
    """
    Combine multiple aberrations.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - coefficients: Dictionary of aberration coefficients.
    
    Returns:
    - Total phase error (in radians).
    """
    phase_error = 0.0
    phase_error += defocus_aberration(kx, ky, coefficients['defocus'])
    phase_error += spherical_aberration(kx, ky, coefficients['spherical'])
    phase_error += coma_aberration(kx, ky, coefficients['coma'])
    phase_error += astigmatism_aberration(kx, ky, coefficients['astigmatism'])
    return phase_error
    
# Amplitude Profiles 
def uniform_amplitude(kx, ky):
    """
    Uniform amplitude profile: constant intensity across the lens.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    
    Returns:
    - Amplitude (constant value of 1).
    """
    return np.ones_like(kx)

def gaussian_amplitude(kx, ky, sigma=0.5):
    """
    Gaussian amplitude profile: smooth falloff in intensity.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - sigma: Width of the Gaussian profile.
    
    Returns:
    - Amplitude (Gaussian distribution).
    """
    return np.exp(-(kx**2 + ky**2) / (2 * sigma**2))

def top_hat_amplitude(kx, ky, radius=1.0):
    """
    Top-hat amplitude profile: sharp cutoff at the edges.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - radius: Radius of the lens aperture.
    
    Returns:
    - Amplitude (1 inside the aperture, 0 outside).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.where(r <= radius, 1.0, 0.0)


def apodized_amplitude(kx, ky, sigma=0.5):
    """
    Apodized amplitude profile: smooth tapering at the edges.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - sigma: Width of the apodization profile.
    
    Returns:
    - Amplitude (apodized distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-(r**2) / (2 * sigma**2)) * (1 - r**2)

def ring_amplitude(kx, ky, radius=0.7, width=0.1):
    """
    Ring-shaped amplitude profile: intensity concentrated in a ring.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - radius: Radius of the ring.
    - width: Width of the ring.
    
    Returns:
    - Amplitude (ring-shaped distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-((r - radius)**2) / (2 * width**2))

def absorption_amplitude(kx, ky, absorption_coeff=0.1):
    """
    Absorption-based amplitude profile: gradual decrease in intensity due to material absorption.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - absorption_coeff: Absorption coefficient.
    
    Returns:
    - Amplitude (absorption-based distribution).
    """
    r = np.sqrt(kx**2 + ky**2)
    return np.exp(-absorption_coeff * r)


def combined_amplitude(kx, ky, profiles):
    """
    Combine multiple amplitude profiles.
    
    Parameters:
    - kx, ky: Transverse k-vector components (normalized to pupil coordinates).
    - profiles: List of amplitude profile functions.
    
    Returns:
    - Combined amplitude profile.
    """
    amplitude = 1.0
    for profile in profiles:
        amplitude *= profile(kx, ky)
    return amplitude
