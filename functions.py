import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from math import pi 
import atom_info as af

def reverse_kouts_to_pixels(kouts, intensities, detector_size, pixel_size, detector_distance):
    """
    Reverse map k_out vectors to detector pixel indices.

    Args:
        kouts (np.ndarray): Array of k_out vectors (N, 3), normalized.
        detector_size (tuple): Detector dimensions in meters (width, height).
        pixel_size (tuple): Pixel size in micrometers (width, height).
        detector_distance (float): Distance from the crystal to the center of the detector in meters.

    Returns:
        np.ndarray: Array of pixel indices for k_out vectors.
    """
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nrows, ncols = (detector_size / pixel_size).astype(int)

    # Reverse mapping to pixel indices
    row_pixels = (kouts[:, 0] / kouts[:, 2]) * detector_distance / pixel_size[0] + nrows / 2
    
    col_pixels = (kouts[:, 1] / kouts[:, 2]) * detector_distance / pixel_size[1] + ncols / 2

    # Convert to integer pixel indices using rounding
    row_pixel_indices = np.floor(row_pixels).astype(int)
    col_pixel_indices = np.floor(col_pixels).astype(int)

    # Verify indices are within bounds
    in_bounds_mask = (row_pixel_indices >= 0) & (row_pixel_indices < nrows) & \
                (col_pixel_indices >= 0) & (col_pixel_indices < ncols)

    
    row_pixel_indices = row_pixel_indices[in_bounds_mask]
    col_pixel_indices = col_pixel_indices[in_bounds_mask]

    intensities = intensities[in_bounds_mask]

    return np.vstack((col_pixel_indices, row_pixel_indices)).T, intensities


def generate_detector_image(intensities, kouts, detector_size, pixel_size, distance):
    """
    Generate a 2D detector image based on scattering intensities and kouts, accounting for distance R.

    Args:
        intensities (np.ndarray): Array of corresponding scattering intensities (N,).
        indices: indices of the intensities on the detector 
        detector_size (tuple): Detector dimensions in meters (width, height).
        pixel_size (tuple): Pixel size in micrometers (width, height).
    Returns:
        np.ndarray: 2D detector image with intensities.
    """
    indices, intensities = reverse_kouts_to_pixels(kouts, intensities, detector_size, pixel_size, distance)
    
    # Convert pixel size to meters
    pixel_size = np.array(pixel_size) * 1e-6
    
    # Compute number of pixels in each dimension
    nx, ny = (detector_size / pixel_size).astype(int)

    detector_image = np.zeros((nx,ny))


    np.add.at(detector_image, (indices[:,0], indices[:,1]), intensities)
    
    ## Populate the image array with intensities
    #for i in range(len(intensities)):
    #    detector_image[indices[i,0], indices[i,1]] += intensities[i]

    return detector_image

def compute_norms_chunk(chunk):
    return np.linalg.norm(chunk, axis=1)
    
def filter_elastic_scatt(kouts, kins, tolerance, wavelength):
    """
    
    Filters the kouts and kins to consider only when:
    |kout| = |kin| +/- 1e-4

    Args:
        kouts (_type_): _description_
        kins (_type_): _description_
    """
    
    # Compute the magnitudes of kin and kout
    # kin_magnitudes = np.linalg.norm(kins, axis=1)  
    # kout_magnitudes = np.linalg.norm(kouts, axis=1)  
    
    # Parallelize norm computation
    kout_magnitudes = np.concatenate(Parallel(n_jobs=-16)(delayed(compute_norms_chunk)(chunk) for chunk in np.array_split(kouts, 8)))

        
    magnitude = 2*np.pi / wavelength
    
    magnitude_diff = np.abs(magnitude - kout_magnitudes)
    mask = magnitude_diff < tolerance
        
    # Apply the mask to get the filtered kin and kout pairs
    if kins.shape[0] > 1:
        filtered_kin = kins[mask]
    else: 
        filtered_kin = kins
        
    filtered_kout = kouts[mask]
    
    return filtered_kout, filtered_kin, mask


def calculate_scattering_amplitude_chunked(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a, shift = None, mask_Ri=None, n_jobs=8, chunk_size=10000):
    """
    Chunked version of scattering amplitude calculation.
    Processes q_vec in chunks to limit memory usage.
    Returns the concatenated result of all chunks.
    """
    N = q_vec.shape[0]
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
    
    results = []

    for start, end in tqdm(chunks, desc="Scattering amplitude chunks"):
        q_chunk = q_vec[start:end]

        form_factor = calculate_form_factor(real_lattice_vecs, q_chunk, R_i, mask_Ri, n_jobs=n_jobs)
        structure_factor = calculate_structure_factor(atoms, rj_atoms, q_chunk, wavelength_a,n_jobs=n_jobs)

        scattering_amp_chunk = form_factor * structure_factor

        if shift is not None:
            phase_shift = compute_phase_shift(q_chunk, shift, n_jobs=n_jobs)
            scattering_amp_chunk *= phase_shift
        results.append(scattering_amp_chunk)

    return np.concatenate(results)

def calculate_atomic_formfactor(atom: str, qvec: np.ndarray, wavelength_a: float):

    """
    Calculates the atomic form factor for a given Q vector and wavelength.

    """
    theta = qmag2ttheta(np.linalg.norm(qvec, axis=-1), wavelength_a= wavelength_a)

    s = np.sin(theta ) / (wavelength_a)

    abc = af.formfactor[atom]
        
    a1 = abc[0]
    a2 = abc[1]
    a3 = abc[2]
    a4 = abc[3]
    b1 = abc[4]
    b2 = abc[5]
    b3 = abc[6]
    b4 = abc[7]
    c = abc[8]

    Z = abc[-1]

    inter = Z - 41.78214 * s**2  * ( (a1*np.exp(-b1*s**2)) + a2*np.exp(-b2*s**2) + a3*np.exp(-b3*s**2) +  a4*np.exp(-b4*s**2) ) +c
    return inter

def calculate_form_factor(real_lattice_vecs, q_vec, R_i, mask_Ri= None, n_jobs = 8):
    
    """
    Input:
        real_lattice_vecs (np.ndarray): 3x3 array containing the real space lattice vectors
        q_vec (np.ndarray): The reciprocal lattice vectors to consider
        R_i (np.ndarray): the real space unit cell positions
        
        Optional:

            mask_Ri (np.ndarray): Mask Ris outside the region of interest (for ptycho scan)
        
    Returns: 
        f_q (np.ndarray): the Form factor of the crystal
    
    Following scattering from atoms in a crystal
    The form factor is Sum_{Unit Cells in Lattice} exp[-i(q.R_i)]
    R_i is the unit cell position
    """
    
    # Volume of unit cell
    V_cell = np.dot(real_lattice_vecs[0], np.cross(real_lattice_vecs[1], real_lattice_vecs[2])) 
    
    if mask_Ri is not None:
        R_i = R_i * mask_Ri[:,np.newaxis]
    
    #phase = np.exp(-1j * np.einsum('ij,kj->ik', q_vec, R_i))
    phase = compute_phase_parallel(q_vec, R_i, batch_size=10000, n_jobs=n_jobs)

    #phase = np.exp(-1j*np.dot(q_vec,R_i.T))

    #scattering_strength_coeff = (np.sum(mask_Ri) / mask_Ri.shape[0]) # FIX ME!
    
    f_q = 2*pi * np.sum(phase, axis = -1) / V_cell #* scattering_strength_coeff
    
    return f_q

def compute_phase_batch(q_batch, rj):
    """Computes a batch of phase values."""
    return np.exp(-1j * np.einsum('ij,kj->ik', q_batch, rj))
    
def compute_phase_parallel(q_vec, rj, batch_size=1000, n_jobs=4):
    """
    Compute phase = exp(-1j * dot(q_vec, rj.T)) using joblib for parallel processing.

    Args:
        q_vec (ndarray): (N, 3) wavevectors.
        rj (ndarray): (M, 3) atomic positions.
        batch_size (int): Number of rows in q_vec to process per batch.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        ndarray: (N, M) phase matrix computed in parallel.
    """
    N = q_vec.shape[0]

    # Define batch ranges
    batch_ranges = [(i, min(i + batch_size, N)) for i in range(0, N, batch_size)]

    # Run parallel processing using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_phase_batch)(q_vec[start:end], rj) for start, end in batch_ranges
    )

    # Merge results into a single array
    phase = np.concatenate(results)
    return phase
    
def calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a,n_jobs=8):
    """
    The structure factor at q_vec, followiung s_q = Sum_{atoms in unit cell} f_j * exp[-1j* q.r_j]

    Input: 
        atoms (list): list of the atom names
        rj_atoms (list or np.ndarray): the locations of the atoms in the unit cell, in the same order as atoms
        q_vec: the q vectors of the experiment
        wavelength_a: the wavelength in Angs
        
    Returns:
        np.ndarray: Structure factor for every q_vec
    """
    rj = np.array(rj_atoms)
    fjs = []
    for atom in atoms:
        fj = calculate_atomic_formfactor(atom, q_vec, wavelength_a)
        fjs.append(fj)
        
    fjs = np.array(fjs)
    
    phase = compute_phase_parallel(q_vec, rj, batch_size=10000, n_jobs=n_jobs)

    #phase = np.exp(-1j * np.einsum('ij,kj->ik', q_vec, rj))
    #phase = np.exp(-1j*np.dot(q_vec,rj.T))
 
    t = fjs.T*phase
    s_q = np.sum(t, axis = -1)
    return s_q

def calculate_scattering_amplitude(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a, phase_shift = None, mask_Ri = None, n_jobs = 8):
    
    """
    Calculates the scattering amplitudes F(q) = S(q) [the structure factor] * f(q) [The form factor]
    
    Input: 
        real_lattice_vecs (np.ndarray): 3x3 array containing the real space lattice vectors
        q_vec (np.ndarray): The reciprocal lattice vectors to consider
        R_i (np.ndarray): the real space unit cell positions
        atoms (list): list of the atom names
        rj_atoms (list or np.ndarray): the locations of the atoms in the unit cell, in the same order as atoms
        wavelength_a: the wavelength in Angs

    Returns:
        np.ndarray: the scattering amplitude
    """
    form_factor = calculate_form_factor(real_lattice_vecs, q_vec, R_i, mask_Ri, n_jobs=n_jobs)
    structure_factor = calculate_structure_factor(atoms, rj_atoms, q_vec, wavelength_a)
    scattering_amp = form_factor*structure_factor

    
    return scattering_amp

def calculate_scattering_amplitude_chunked(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a, shift = None, mask_Ri=None, n_jobs=8, chunk_size=10000):
    """
    Chunked version of scattering amplitude calculation.
    Processes q_vec in chunks to limit memory usage.
    Returns the concatenated result of all chunks.
    """
    N = q_vec.shape[0]
    chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
    
    results = []

    for start, end in tqdm(chunks, desc="Scattering amplitude chunks"):
        q_chunk = q_vec[start:end]

        form_factor = calculate_form_factor(real_lattice_vecs, q_chunk, R_i, mask_Ri, n_jobs=n_jobs)
        structure_factor = calculate_structure_factor(atoms, rj_atoms, q_chunk, wavelength_a,n_jobs=n_jobs)

        scattering_amp_chunk = form_factor * structure_factor

        if shift is not None:
            phase_shift = compute_phase_shift(q_chunk, shift, n_jobs=n_jobs)
            scattering_amp_chunk *= phase_shift
        results.append(scattering_amp_chunk)

    return np.concatenate(results)


# -------------------------
# Utilities: safe batched phase without storing NxM arrays
# -------------------------
def structure_factor_batched(atoms, rj_atoms, q_vec, wavelength_a, batch_size=10000, n_jobs=1):
    """
    Compute structure factor S(q) = sum_j f_j(q) * exp(-i q·r_j)
    Processes q_vec in batches so we never build a full (Nq x Natoms) array.
    - atoms: list of atom names
    - rj_atoms: (Natoms, 3)
    - q_vec: (Nq, 3)
    """
    rj = np.array(rj_atoms)          # (Natoms, 3)
    Natoms = rj.shape[0]
    Nq = q_vec.shape[0]

    # Precompute atomic form factors per atom for each q batch (memory friendly)
    def worker(q_batch):
        # q_batch: (B,3)
        # compute f_j(q) for each atom j -> shape (Natoms, B) ideally compute per atom
        # but since calculate_atomic_formfactor accepts q-array we use it per atom
        fjs = []
        for atom in atoms:
            fj = calculate_atomic_formfactor(atom, q_batch, wavelength_a)  # shape (B,)
            fjs.append(fj)
        fjs = np.array(fjs)            # (Natoms, B)
        # compute phase: exp(-1j * q_batch @ rj.T) -> shape (B, Natoms)
        phase = np.exp(-1j * np.dot(q_batch, rj.T))    # (B, Natoms)
        # multiply and sum across atoms -> (B,)
        s_q_batch = np.sum(phase * fjs.T, axis=1)
        return s_q_batch

    # iterate in batches
    batches = [(i, min(i + batch_size, Nq)) for i in range(0, Nq, batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(q_vec[s:e]) for s, e in batches
    )
    return np.concatenate(results)


# -------------------------
# Form-factor: lattice sum, chunked and BLAS-friendly
# -------------------------
def calculate_form_factor_chunked(real_lattice_vecs, q_vec, R_i, mask_Ri=None, batch_size=5000, n_jobs=1):
    """
    Compute f(q) = sum_R exp(-i q·R) for many q values, using batching.
    real_lattice_vecs: 3x3
    q_vec: (Nq,3)
    R_i: (N_R, 3) array of lattice positions (already expanded by crystal.crystal_size)
    mask_Ri: optional boolean mask to limit to illuminated subset
    """
    if mask_Ri is not None:
        # keep only positions where mask true; mask_Ri should be (N_R,)
        R = R_i[mask_Ri]
    else:
        R = R_i
    Nq = q_vec.shape[0]
    batches = [(i, min(i + batch_size, Nq)) for i in range(0, Nq, batch_size)]

    def worker(q_batch):
        # q_batch: (B,3)
        # compute dot -> (B, N_R)
        dot = np.dot(q_batch, R.T)              # (B, N_R)
        # complex exponential and sum over axis 1 -> (B,)
        return np.sum(np.exp(-1j * dot), axis=1)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(q_vec[s:e]) for s, e in batches
    )
    f_q = np.concatenate(results)
    # normalization as you used before
    V_cell = np.dot(real_lattice_vecs[0], np.cross(real_lattice_vecs[1], real_lattice_vecs[2]))
    f_q = 2*pi * f_q / V_cell
    return f_q


# -------------------------
# Replace compute_phase_parallel (which returns huge matrix) with batched compute 
# -------------------------
# Improved scattering amplitude chunked (single function using the above helpers)
# -------------------------

def calculate_scattering_amplitude_chunked_v2(real_lattice_vecs, q_vec, R_i, atoms, rj_atoms, wavelength_a,
                                             mask_Ri=None, n_jobs=8, q_batch_size=8000):
    """
    Combined chunked calculation that avoids building Nq x N_R or Nq x Natom matrices in memory.
    """
    Nq = q_vec.shape[0]
    batches = [(i, min(i + q_batch_size, Nq)) for i in range(0, Nq, q_batch_size)]
    results = []

    for start, end in tqdm(batches, desc="scattering chunks"):
        q_batch = q_vec[start:end]  # (B,3)
        # form factor for q_batch
        f_q_batch = calculate_form_factor_chunked(real_lattice_vecs, q_batch, R_i, mask_Ri=mask_Ri,
                                                  batch_size=1000, n_jobs=n_jobs)
        # structure factor for q_batch
        s_q_batch = structure_factor_batched(atoms, rj_atoms, q_batch, wavelength_a,
                                             batch_size=1000, n_jobs=n_jobs)
        results.append(f_q_batch * s_q_batch)
    return np.concatenate(results)
# ----------------------------------------------------------------------------------------------------

def qmag2ttheta(qmag: float, **kwargs) -> float:

    """Calculates the corresponding two theta value for a given Q vector magnitude and a particular X-ray energy. 

    Args:
        qmag: Magnitude of Qvector 
        energy_kev: energy of the xrays [keV]
        wavelength: Xray wavelenght [A]

    Returns:
        ttheta: Two Theta Value [Degrees]
    """
    
    if "energy_kev" in kwargs:
        energy_kev = kwargs["energy_kev"]
        wavelength_a = energy2wavelength_a(energy_kev)  # wavelength form photon energy
        
    elif "wavelength_a" in kwargs:
        wavelength_a = kwargs["wavelength_a"]
        
    else:
        raise ValueError("Supply either a energy (energy_kev) or wavelength (wavelength_a)")
        

    theta = np.arcsin(qmag * wavelength_a/(4*np.pi))

    theta = np.rad2deg(theta)
    ttheta = 2*theta

    return ttheta

def compute_phase_shift(q_vec, shift, batch_size=10000, n_jobs=8):
    """
    Computes the phase shift due to a crystal shift relative to the beam. 
    """
    phase = compute_phase_parallel(q_vec, shift, batch_size=1000, n_jobs=n_jobs)

    return phase