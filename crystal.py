

import numpy as np
from math import sin, cos, pi
import atom_info as af 


class Crystal:
    
    def __init__(self):
        
        self._uc_size = None
        self._uc_shape = None
        self.real_space_vectors = None
        self._recip_space_lattice= None
        self._ref_idx = None
        self._crys_size = None
        self._shape_array = None
        self._max_hkl = None
        self._orientation = None
    
        self.loc_atoms = {}
        self.abc_atoms = {}
        self.atoms_list = []
        self.atom_counter = {}
        
        self.rlvs = None
        self.gvectors = None
        self.object = None
        self.flatobject = None
        self.real_space_coords = None
    
    @property
    def unit_cell_size(self):
        """
        The size of the unit cell in Angs
        
        np.ndarray representing the size of the unit cell
        """
        return self._uc_size
    
    @unit_cell_size.setter
    def unit_cell_size(self, size: tuple) -> None:
        """
        Set the size of the unit cell in Angs.

        Input: (a,b,c) tuple representing the sizes of the unit cell
        Output: None
        """
        self._uc_size = np.array(size)
        
    @property
    def unit_cell_shape(self):
        """
        The Shape of the unit cell [degrees]
        
        [alpha, beta, gamma] array representing the angles of the unit cell. 

        """
        return self._uc_shape
    
    @unit_cell_shape.setter
    def unit_cell_shape(self, angles:tuple)->None:
        """
        Set the Shape of the unit cell [degrees]

        Input: 
            angles (tuple): (alpha, beta, gamma) tuple representing the angles of the unit cell. 
        
        Output: None
        """
        self._uc_shape = np.array(angles)
        
    @property
    def crystal_size(self):
        """The Size of the whole crystal

        [x,y,z] array representing the dimensions of the crystal
            
        """
        return self._crys_size
    
    @crystal_size.setter
    def crystal_size(self, size:tuple):
        """
        
        Set The Size of the whole crystal
        Temporary: for now this is the number of unit cells in each dimension
        
        Args:
            size (tuple): (x,y,z) tuple representing the dimensions of the crystal
            
        """
        self._crys_size = np.array(size)
    
    
    def set_real_lattice_vectors(self):
        """
        The real space unit cell vectors for the crystal lattice. 
        """   
        if self.real_space_vectors is None:
            self.real_space_vectors = calc_realspace_lattice_vectors(self.unit_cell_size, self.unit_cell_shape)
        
        return self.real_space_vectors
    
    @property
    def crystal_orientation(self):
        
        return self._orientation
    
    @crystal_orientation.setter
    def crystal_orientation(self, orienation:tuple):
        
        self._orientation = orienation
        
    def rotate_UC(self):
        
        matrix = StanRotMat(*self.crystal_orientation)
        
        self.real_space_vectors = np.dot(matrix, self.real_space_vectors)
    
    def set_recip_lattice_vectors(self):
        
        """
        The reciprocal Latttice vectors of the crystal at study
        """
        
        if self._recip_space_lattice is None:
            
            self._recip_space_lattice = calc_reciprocal_lattice_vectors(self.real_space_vectors)
            
        return self._recip_space_lattice
    
    def add_atom(self, atom: str, location: list):
        """Add atom name and positions in the unit cell in fractional coordinates
        
        This then extracts the variables for computing the atomic form factor, i.e. a,b,c,Z
        
        Args: 
            atom: String of the elemental atom name
            location: List of floats representing the fractional coordinates of the atom
        """
        
        loc = np.dot(self.real_space_vectors,location) 
        
        abc_vals = af.formfactor[atom]
        
        if atom not in self.atom_counter:
            self.atom_counter[atom] = 0
        self.atom_counter[atom] += 1
        
        atom_name = f"{atom}{self.atom_counter[atom]}"
        
        self.atoms_list.append(atom)

        self.loc_atoms[atom_name] = loc
        
        self.abc_atoms[atom_name] = abc_vals + [af.elements_z[atom]]
    
    @property
    def refractive_index(self):
        """
        The refractive index of the material
        """
        return self._ref_idx

    @refractive_index.setter
    def refractive_index(self, n):
        """
        Args:
            n (float): The refractive index of the material
        """
        self._ref_idx = n
        
    @property
    def max_hkl(self):
        """
        The refractive index of the material
        """
        return self._max_hkl

    @max_hkl.setter
    def max_hkl(self, hkl):
        """
        Args:
            n (float): The refractive index of the material
        """
        self._max_hkl = hkl
        
    def gen_RLS(self, grid_points, ravel=False, threshold = 0.2):
        
        qvecs = gen_RLS_from_maxhkl_maskOrigin(self.max_hkl, grid_points, *self._recip_space_lattice, threshold= threshold, ravel=ravel)
        
        self.rlvs = qvecs
        
    def gen_RLS_one_hkl(self, grid_points, hkl, range_scale, ravel=True):
        
        
        qvecs, _ = gen_rlvs_for_one_hkl(hkl=hkl, recip_vecs=self._recip_space_lattice, grid_points=grid_points, range_scale=range_scale, ravel=ravel)
        
        #qvecs = generate_recip_lattice_points_hkl(self._recip_space_lattice, hkl, range_scale, grid_points)
        
        self.rlvs = qvecs
        
    def gen_Gs(self):
        
        gs =  generate_recip_lattice_points(self._recip_space_lattice,self.max_hkl)
        
        self.gvectors = gs
    
    @property
    def q_max(self):
        
        hmax = (self.max_hkl / self.unit_cell_size[0])**2
        kmax = (self.max_hkl / self.unit_cell_size[1])**2
        lmax = (self.max_hkl / self.unit_cell_size[2])**2
        
        return 2 * np.pi * np.sqrt(hmax+kmax+lmax)
    
    
    def set_shape_array(self, normals, padding = (2,2,2)):
        """
        Create a shape function array representing the crystal's geometry.

        Args:
            arraysize (tuple): Dimensions of the 3D array representing the crystal grid.
            normals (list): List of facet normal vectors. Each facet is defined by a set of
                            start and end coordinates [x0, y0, z0, x1, y1, z1].

        Returns:
            None: Updates the `shape_array` and `real_space_coords` attributes.
        """
        self.crys_indices, self.shape_array = set_shape_array(self.crystal_size, normals, padding)
        self.real_space_coords = np.dot(self.crys_indices, self.real_space_vectors) # Real Space Coordinates of every unit cell in the lattice
        
        
    def set_illuminated_volume(self, interaction_volume=(0,2,0,2,0,9)):
        """      
        A function that masks most of the crystal by masking real space coordinates array in an interaction volume.  
        The interaction volume can then be changed to simulate scanning over the crystal in a ptychographic manner
        
        Args:                                
            interaction_volume (callable, optional): (6,) array 
                                                     (xi,xf,yi,yf,zi,zf)
                                                     where the elements describe the range of unit cell indices in the interaction volume. 
        """
        
        self.illuminated_coords, self.illuminated_indices = define_interaction_volume(*interaction_volume, self.real_space_vectors)

############################# Functions #############################
def calc_realspace_lattice_vectors(uc_size, uc_angles) -> np.ndarray:
    """
    Calculates the unit cell vectors for the crystal lattice. 

    input: 
        - uc_size: [a,b,c] A list of the size of the unit cell in Angstroms.
        - uc_angles: [alpha, beta, gamma] A list of the angles between the real space vectors in degrees. 

    output: A numpy array of the real space lattice vectors.
    """

    # Define primittive lattice vectors
    x = np.array([1,0,0])
    y= np.array([0,1,0])
    z = np.array([0,0,1])


    alpha = np.deg2rad(uc_angles[0])
    beta = np.deg2rad(uc_angles[1])
    gamma = np.deg2rad(uc_angles[2])

    a = uc_size[0]
    b = uc_size[1]
    c = uc_size[2]

    # Calculate lattice vector directions
    avec = a * x 
    bvec = b * x * cos(gamma)  + b * y * sin(gamma)
    cvec = c * cos(beta) * x + c * ((cos(alpha) - cos(beta)*cos(gamma))/ sin(gamma)) * y + z* c * np.sqrt( 1 - cos(beta)**2 - ((cos(alpha) - cos(beta)*cos(gamma))/ sin(gamma))**2)

    realspaceVecs = np.array([avec,bvec,cvec])

    return realspaceVecs

def calc_reciprocal_lattice_vectors(realspaceVecs: np.ndarray) -> np.ndarray:
    """
    Calculate the reciprocal lattice vectors from the real space vectors. 

    input: 
        - realspaceVecs: An array of the real space lattice vectors directions [A]. 

    output: A numpy array of the reciprocal space lattice vectors. 
    """

    avec = realspaceVecs[0]
    bvec = realspaceVecs[1]
    cvec = realspaceVecs[2]

    # Calculate Reciprocal lattice vectors
    a_star =  2*pi * np.cross(bvec, cvec) / (np.dot(avec, np.cross(bvec,cvec)))
    b_star =  2*pi * np.cross(cvec, avec) / (np.dot(bvec, np.cross(cvec,avec)))
    c_star =  2*pi * np.cross(avec, bvec) / (np.dot(cvec, np.cross(avec,bvec)))

    recpspaceVecs = np.array([a_star, b_star, c_star])

    return recpspaceVecs

def gen_RLS_from_maxhkl_maskOrigin(max_hkl, grid_points, a, b, c, threshold=0.1, ravel=False):
    """
    Generate reciprocal space q-vectors for a general lattice and remove the 0,0,0 order and surrounding cube.

    Parameters:
        max_hkl (int): Maximum value of h, k, l indices.
        grid_points (int): Number of grid points along each axis.
        a, b, c (np.ndarray): Reciprocal-space lattice vectors.
        threshold (float): Distance threshold to exclude points around (0, 0, 0).
        ravel (bool): Whether to return the q-vectors as a flat array.

    Returns:
        q_vectors (numpy.ndarray): Array of q-vectors with the 0,0,0 region removed.
    """
    # Reciprocal lattice basis vector magnitudes
    b1 = np.linalg.norm(a)
    b2 = np.linalg.norm(b)
    b3 = np.linalg.norm(c)
        
    # Define h, k, l ranges
    h = np.linspace(-max_hkl, max_hkl, grid_points) 
    k = np.linspace(-max_hkl, max_hkl, grid_points) 
    l = np.linspace(-max_hkl, max_hkl, grid_points) 

    # Generate 3D grid of q-space
    h_grid, k_grid, l_grid = np.meshgrid(h, k, l, indexing="ij")
    
    # Calculate qx, qy, qz components
    qx = h_grid * b1 
    qy = k_grid * b2
    qz = l_grid * b3

    # Combine components into a single array
    q_vectors = np.stack((qx, qy, qz), axis=-1)
    
    # Compute the distance of each point from the origin
    distances = np.sqrt(qx**2 + qy**2 + qz**2)

    # Mask out the region near the origin (including 0,0,0)
    mask = distances > threshold

    if ravel:
        # Stack into a single array of shape (grid_points**3, 3)
        q_vectors = q_vectors[mask]
        
    else: 
        q_vectors[~mask] = np.nan 

    return q_vectors

def gen_rlvs_for_one_hkl(hkl:tuple, recip_vecs, grid_points= None, range_scale= 0.49, ravel = True):
    
    h,k,l = hkl
    
    G_vec = h*recip_vecs[0] +  k*recip_vecs[1] + l*recip_vecs[2]
    
    # Reciprocal lattice basis vector magnitudes
    b1 = np.linalg.norm(recip_vecs[0])
    b2 = np.linalg.norm(recip_vecs[1])
    b3 = np.linalg.norm(recip_vecs[2])
    
    # Define h, k, l ranges
    h = np.linspace(-1, 1, grid_points) * b1 * range_scale
    k = np.linspace(-1, 1, grid_points) * b2 * range_scale 
    l = np.linspace(-1, 1, grid_points) * b3 * range_scale
    
    # Generate 3D grid of q-space
    h_grid, k_grid, l_grid = np.meshgrid(h, k, l, indexing="ij")
    
    # Calculate qx, qy, qz components
    qx = h_grid 
    qy = k_grid 
    qz = l_grid 
    
    # Combine components into a single array
    q_vectors = G_vec + np.stack((qx, qy, qz), axis=-1)
    
    if ravel:
        q_vectors = q_vectors.reshape(-1,3)
    return q_vectors, (h,k,l)

def generate_recip_lattice_points(recpspaceVecs: np.ndarray, max_hkl: int) -> np.ndarray:

    """
    Generates a set of reciprocal lattice points 

    Input: 
        - recpspaceVecs: A numpy array of the reciprocal space vectors of the system [A^-1]
        - max_hkl: The maximum Miller index value to generate

    Returns:
        H_hkl: A numpy array containing a set of reciprocal lattice points. 
    """
    h_range = range(-max_hkl, max_hkl + 1)
    l_range = range(-max_hkl, max_hkl + 1)
    k_range = range(-max_hkl, max_hkl + 1)
    
    H_hkl = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                if not (h == 0 and k == 0 and l == 0):
                #if h!=0 and k!=0 and l!=0:
                    H = h * recpspaceVecs[0] + k * recpspaceVecs[1] + l * recpspaceVecs[2]
                    H_hkl.append(H)
    
    H_hkl = np.array(H_hkl)
    
    return H_hkl

def define_interaction_volume(xi, xf, yi, yf, zi, zf, lattice_vectors):
    """
    Define the interaction volume in lattice coordinates based on unit cell bounds.

    Parameters:
    -----------
    xi, xf : int
        Bounds along the a-axis (unit cell indices).
    yi, yf : int
        Bounds along the b-axis (unit cell indices).
    zi, zf : int
        Bounds along the c-axis (unit cell indices).
    a, b, c : np.ndarray
        Lattice vectors (each of shape (3,)).

    Returns:
    --------
    np.ndarray
        Array of shape (N, 3), where N is the number of lattice points in the interaction volume.
    """
    # Generate all combinations of indices within the bounds
    i_indices = np.arange(xi, xf )
    j_indices = np.arange(yi, yf )
    k_indices = np.arange(zi, zf )
    
    # Create a grid of all combinations (i, j, k)
    i, j, k = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')
    
    indices = np.stack((i.ravel(), j.ravel(), k.ravel()), axis = 1)  # Shape: (N, 3)

    # Project indices onto the lattice
    lattice_coords = np.dot(indices, lattice_vectors)
    return lattice_coords, indices

##################### Geometry ###################
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




#################### Crystal Shape ##################

def set_shape_array(arraysize, normals, padding= (2,2,2)):
    """
    Create a shape function array representing the crystal's geometry.

    Args:
        arraysize (tuple): Dimensions of the 3D array representing the crystal grid.
        normals (list): List of facet normal vectors. Each facet is defined by a set of
                        start and end coordinates [x0, y0, z0, x1, y1, z1].

    Returns:
        None: The indices, and the shape array
    """
    
    # Apply padding to array size
    padded_size = (arraysize[0] + 2 * padding[0],
                   arraysize[1] + 2 * padding[1],
                   arraysize[2] + 2 * padding[2])
    
    # Initialize the shape array
    shape_array = np.zeros(arraysize, dtype=np.uint8)  # Binary array to represent shape (0 or 1)
    
    
    # n_voxels = np.prod(arraysize)
    
    # # Generate 3D grid indices for the shape array
    # indices = np.transpose(np.unravel_index(np.arange(n_voxels), arraysize))
    
    indices = np.stack(np.meshgrid(np.arange(arraysize[0]),
                                   np.arange(arraysize[1]),
                                   np.arange(arraysize[2]),
                                   indexing='ij'), -1).reshape(-1, 3)
    
    # Process each surface defined by normals
    for surface in normals:
        start = np.array(surface[:3])  # Start coordinates
        end = np.array(surface[3:])   # End coordinates
        normal = (end - start)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector

        # Determine which points lie "inside" the surface
        inside = np.dot(indices - start, normal) < 0
    
        
        shape_array[tuple(indices[inside].T)] = 1
        #flat_shape_array[inside] = 1  # Mark voxels inside the shape as 1s

    # Store the shape array
    shape_array = np.ascontiguousarray(shape_array, dtype=np.uint8)

    shape_array = pad(shape_array,padding[0], padding[0],
                                  padding[1], padding[1],
                                  padding[2], padding[2])
    
    # Adjust indices to reflect padding shift
    padded_indices = indices + np.array(padding)  # Shift indices to match padded space

    return padded_indices, shape_array

def cuboid_normals(arraysize):
    
    X=arraysize[0]
    Y=arraysize[2]
    Z=arraysize[2]
    dX = 30
    X2=X/2 - dX
    Y2=Y/2 - dX
    Z2=Z/2 - dX
    s3 = np.sqrt(3)
    
    normals = [
    [dX,Y2,Z2,dX-1,Y2,Z2],\
    [X-dX,Y2,Z2,X,Y2,Z2],\
    [X2/2+X/2,s3*Y2/2+Y/2,Z2,X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
    [X2/2+X/2,-s3*Y2/2+Y/2,Z2,X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
    [-X2/2+X/2,s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
    [-X2/2+X/2,-s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
    [X2,Y2,dX,X2,Y2,dX-1],\
    [X2,Y2,Z-dX,X2,Y2,Z]
    ]
    
    return normals

def hexagonal_normals(arraysize):
    """
    Generate normal vectors for a hexagonal prism.

    Args:
        arraysize (tuple): (X, Y, Z) dimensions of the crystal.

    Returns:
        list: Normals defining the hexagonal prism.
    """
    X, Y, Z = arraysize
    dX = 30  # Buffer distance from edges
    dZ = 30  # Height buffer
    X2 = X / 2 - dX  # Distance from center to hexagon vertices
    Y2 = Y / 2 - dX
    Z2 = Z - dZ  # Top and bottom face Z-coordinates

    # Compute hexagonal vertices in the XY plane
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (hexagon)
    hexagon_vertices = np.array([[X2 * np.cos(a) + X / 2, Y2 * np.sin(a) + Y / 2] for a in angles])

    normals = []

    # Top and bottom hexagonal faces
    for v in hexagon_vertices:
        normals.append([v[0], v[1], dZ, v[0], v[1], dZ - 1])  # Bottom face
        normals.append([v[0], v[1], Z2, v[0], v[1], Z2 + 1])  # Top face

    # Side faces (connect top and bottom hexagons)
    for i in range(6):
        x0, y0 = hexagon_vertices[i]
        x1, y1 = hexagon_vertices[(i + 1) % 6]  # Next vertex in sequence
        normals.append([x0, y0, dZ, x1, y1, dZ])  # Bottom edge
        normals.append([x0, y0, Z2, x1, y1, Z2])  # Top edge

    return normals

def pad(array, psx, psy, psz, pex, pey, pez):

    dtype = array.dtype
    
    shp = array.shape
    array = np.concatenate((np.zeros((psx,shp[1],shp[2]), dtype = dtype, order = 'C'), array), axis = 0)
    array = np.concatenate((array, np.zeros((pex,shp[1],shp[2]), dtype = dtype, order = 'C')), axis = 0)
    shp = array.shape
    array = np.concatenate((np.zeros((shp[0],psy,shp[2]), dtype = dtype, order = 'C'), array), axis = 1)
    array = np.concatenate((array, np.zeros((shp[0],pey,shp[2]), dtype = dtype, order = 'C')), axis = 1)
    shp = array.shape
    array = np.concatenate((np.zeros((shp[0],shp[1],psz), dtype = dtype, order = 'C'), array), axis = 2)
    array = np.concatenate((array, np.zeros((shp[0],shp[1],pez), dtype = dtype, order = 'C')), axis = 2)

    return array
    
if __name__ == '__main__':  
    crys = Crystal()

    crys.unit_cell_shape = (90,90,90) 
    crys.unit_cell_size = (1,1,1)
    crys.crystal_size = (10,10,10)
    
    crys.set_real_lattice_vectors()
    crys.crystal_orientation = (45,0,0,0)
    crys.rotate_UC()
    crys.set_recip_lattice_vectors()
    
    cubenormals = cuboid_normals(crys.crystal_size)
    crys.set_shape_array(cubenormals)
    
    crys.add_atom('Si', [.5,.5,.5])
    
    crys.set_illuminated_volume((0,8,0,8,0,3))
    
    mask = crys.is_inisde_crystal(crys.illuminated_indices, crys.crys_indices)

