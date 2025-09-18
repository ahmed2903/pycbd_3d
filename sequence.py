
from curses import meta
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
#import pyvista as pv
from time import strftime
from tqdm import tqdm

sys.path.append(os.path.abspath('/home/mohahmed/CBD_sim/CDIStudio/'))

from plot import plot_3d_array
from utils import indexQ

from crystal import *

from beam import *




def simulate_by_convolution(crystal, beam, detector, plot_2d_st = False, plot_3d_st = False, kin_kout_mapping = False, save_parameters = True, **kwargs):
    """
    Inputs:
        crystal: Crystal Object
        Beam: Beam Object
        Detector: Detector Object
        
        atoms (dict): A dictionary of atoms names and their fractional coordinate in the unit cell.
         
        plot_2d_RL: A boolean value to plot a slice of the 2D samlpes k-space (the shape transform) - Need to then provide a 'slice_index'.
        plot_3d_RL: A boolean value to plot the sampled 3D k-space.
        kin_kout_mapping: A boolean value to plot the kin kout map
        save_parameters: A boolean value to save the parameters of the simulation in a text file. 
        
        shape_normals (np.ndarray): an array representing the normals of the shape of the crystal.
        
        crystal_orientation (optional) (tuple): A tuple representing the crystal orientation (chi, mu, eta, phi).
        
        hkl (tuple): The hkl indices of interest. 
        
        grid_points (int): The number of points in each dimension in the reciprocal space sampling
        
        range_scale (float): fractional range of k-sapce to sample, from the hkl of interest to the neighbouring hkls
                
        low_pass_filter (float): A fractional value of the maximum intensity below which the diffraction intensities are masked
        
        image_scale (float): if passed, the detector image is scaled as image^(image_scale)   
        
        save_folder (string): path to the folder to save simulation results in 
    """
    show = kwargs['show']
    
    if "save_folder" in kwargs:
        path = kwargs["save_folder"]
    
    else:
        path = os.getcwd()
    
    start_time = time.time()
    
    time_str = strftime("%Y-%m-%d_%H.%M")
    fldr_name = f'{path}/simulation_{time_str}'
    
    if fldr_name.split("/")[-1] in os.listdir(path):
        fldr_name += "_1"
    os.mkdir(fldr_name)
    
    crystal.set_real_lattice_vectors()
    
    if 'crystal_orientation' in kwargs:
        crystal.crystal_orientation = kwargs['crystal_orientation']
        crystal.rotate_UC()
        
    crystal.set_recip_lattice_vectors()
    
    shape_normals = kwargs["shape_normals"]
    crystal.set_shape_array(shape_normals)
    
    atoms = kwargs['atoms']
    for key, value in atoms.items():
        crystal.add_atom(key, value)
    
    if 'hkl' not in kwargs:
        raise ValueError("A tuple for the hkl indices to simulate must be provided.")
    
    h,k,l = kwargs['hkl']
    
    if "grid_points" not in kwargs:
        raise ValueError(" grid points value must be given")
    
    grid_points = kwargs["grid_points"]
    
    if "range_scale" in kwargs:
        range_scale = kwargs['range_scale']
    else: 
        range_scale = .5    
    
    #crystal.gen_RLS_one_hkl(grid_points = grid_points, hkl= kwargs['hkl'], range_scale = range_scale, ravel=True)
    crystal.gen_Gs()
    
    hkl = np.array([kwargs["hkl"]])
    
    crystal.rlvs = np.dot(hkl, crystal._recip_space_lattice)
    
    #shape_transform = compute_fourier_transform(crystal.shape_array)
    
    qvecs_tmp, qs = gen_rlvs_for_one_hkl(hkl=(0,0,0), recip_vecs=crystal._recip_space_lattice, grid_points=grid_points,range_scale=range_scale, ravel=False)

    print(qs[0].shape)
    #shape_transform = np.abs(calculate_form_factor(crystal.real_space_vectors, qvecs_tmp, crystal.real_space_coords ))
    shape_transform = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, qvecs_tmp, crystal.real_space_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength ))**2
    
    qmags = np.linalg.norm(qvecs_tmp, axis = 1)
    
    kmin = np.min(qmags)
    kmax = np.max(qmags)
    
    if plot_2d_st:
        reciprocal_lattice_intensity = shape_transform
        if "slice_index" in kwargs:
            slice_index = kwargs["slice_index"]
        else:
            slice_index = grid_points //2
        
        plt.figure(figsize=(8, 8))
        plt.imshow(reciprocal_lattice_intensity[slice_index, :, :])#, extent=(-crystal.q_max, crystal.q_max, -crystal.q_max, crystal.q_max), origin='lower')
        plt.colorbar(label='Intensity')
        plt.title('Reciprocal Lattice (2D Slice)')
        plt.xlabel(r'$q_x$')
        plt.ylabel(r'$q_y$')
        plt.savefig(f"{fldr_name}/2D_RL_slice_{time_str}.jpeg")
        
    if plot_3d_st:
        
        if not plot_2d_st:
            qvecs_tmp = gen_rlvs_for_one_hkl(hkl=kwargs['hkl'], recip_vecs=crystal._recip_space_lattice, grid_points=grid_points,range_scale=range_scale, ravel=False)
        
        #reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, qvecs_tmp, crystal.real_space_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength ))
        reciprocal_lattice_intensity = shape_transform
       
        plot_3d_array(reciprocal_lattice_intensity, show=show, fname = f"{fldr_name}/3D_RL_{time_str}.pdf", opacity = 'linear')
        
        print("3d shape transform plot done")
        
    if "low_pass_filter" in kwargs:
        
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, crystal.rlvs, crystal.real_space_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength )) 
        
        lpf = kwargs["low_pass_filter"] * reciprocal_lattice_intensity.max()
        crystal.rlvs = crystal.rlvs[reciprocal_lattice_intensity>lpf]
    
    # Main computations of k_out
    print("computing kouts")
    kouts, kin_indices, _ = compute_kout_from_G_kin(crystal.rlvs, beam.kins)
    kins = beam.kins[kin_indices]   
    
    print("filtering by direction")    
    max_angle = calc_detector_max_angle(detector.size, detector.distance)
    #kouts, kins = filter_vectors_by_direction(kouts, kins, np.array([0.,0.,1.0]), max_angle)
    
    print("filtering by elasticity")
    if "elasticity_tolerance" in kwargs:
        elasticity_tolerance = kwargs["elasticity_tolerance"]
    else: 
        elasticity_tolerance = 1e-6
        
    kouts, kins = filter_elastic_scatt(kouts, kins, tolerance=elasticity_tolerance, wavelength= beam.wavelength)
    
    qvecs = kouts-kins
    
    print("indexing Q")
    q_indexed = indexQ(qvecs, crystal._recip_space_lattice)
    q_indices = np.unique(q_indexed, axis=0)
        
    #print("calculating scattering amplitude")
    #scat_amp = calculate_scattering_amplitude(crystal.real_space_vectors, qvecs, crystal.real_space_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength )
    ks, its = convolve_reciprocal_lattice_with_grid(shape_transform, kouts, *qs)
    
    print("ks shape")
    print(ks.shape)
    #intensity = calculate_intensity(its)
    
    # Detector image 
    image = generate_detector_image(its, ks, detector.size, detector.pixel_size, detector.distance)
    
    validate_qs = kwargs['validate_qs']    
    
    if validate_qs:
        # Validation
        for i, (kin, kout, qvec) in enumerate(zip(kins, kouts, qvecs)):
            computed_qvec = kout - kin
            angle_ks = calc_angle(kin,kout)
            angle_q = qmag2ttheta(np.linalg.norm(qvec), energy_kev = beam.energy)
            
            assert np.allclose(angle_ks, angle_q), f"Mismatch in angles index {i}: {angle_ks} != {angle_q}"
            assert np.allclose(computed_qvec, qvec), f"Mismatch in vectors at index {i}: {computed_qvec} != {qvec}"
            assert np.allclose(np.linalg.norm(kin), np.linalg.norm(kout)), f"Mismatch in magnitudes at index {i}: {np.linalg.norm(kin)} != {np.linalg.norm(kout)}"
            #print(f"Vector {i} validated: |kin| = {np.linalg.norm(kin):.2f}, |kout| = {np.linalg.norm(kout):.2f}, |qvec| = {np.linalg.norm(qvec):.2f}")


    plt.figure()
    if "image_scale" in kwargs:
        img_scale = kwargs['image_scale']
    else: 
        img_scale = 1
        
    image = image**img_scale
    plt.imshow(image)
    plt.colorbar(label=f'Intensity ^({img_scale})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scattering on Detector')
    plt.show()
    plt.savefig(f"{fldr_name}/detector_image_{time_str}.jpeg")
    
    print("detector image done")
    if kin_kout_mapping:
        #try:
                   
        plotter = pv.Plotter(off_screen=False)
        # Create PyVista PolyData objects  
        #test, _ = convolve_reciprocal_lattice_with_grid(shape_transform, crystal.gvectors, *qs)
        
        cloud1 = pv.PolyData(kins)
        cloud2 = pv.PolyData(ks)

        # Add the point array
        plotter.add_mesh(cloud1, color="red", point_size=1, render_points_as_spheres=True)
        
        plotter.add_mesh(cloud2, color="blue", point_size=1, render_points_as_spheres=True)
        # #plotter.add_mesh(cloud3, color="black", point_size=5, render_points_as_spheres=True )

        plotter.view_xy()
        plotter.background_color = 'white'

        # Add legend and show
        plotter.add_legend([
            ("kins (red)", "red"),
            ("kouts (blue)", "blue"),
        ])

        # Add axes to the plot
        plotter.show_axes()
        plotter.save_graphic(f"{fldr_name}/kin_kout_map_{time_str}.pdf")
        if show:
            plotter.show()
        plotter.close()
        #except:
        #    print("Error with plotting 3D mapping")
        #    pass 
    
    vector_map = kwargs['vector_map']
    if vector_map: 
        
        #Function to add vectors to the plot

        # Create a PyVista plotter
        plotter = pv.Plotter(off_screen=False)

        def add_vectors(plotter, start_points, vectors, color, shaft_scale = 0.25, tip_scale=.2):
            
            magnitude = np.linalg.norm(vectors[0])
            
            shaft_radius = shaft_scale * 1.0 / magnitude 
            tip_radius = shaft_radius+tip_scale
            
            for start, vec in zip(start_points, vectors):
                arrow = pv.Arrow(start=start, direction=vec, scale='auto', shaft_radius=shaft_radius, tip_radius=tip_radius)
                plotter.add_mesh(arrow, color=color)


        # Origin for kins and kouts
        origin = np.array([0, 0, 0])

        # Add kins vectors
        add_vectors(plotter, [origin] * len(kins[:100]), kins[-1000:], "blue", shaft_scale= 0.1, tip_scale=.02)

        # Add kouts vectors
        add_vectors(plotter, [origin] * len(kouts[:100]), kouts[-1000:], "green", shaft_scale= 0.1,tip_scale=.02)

        # Add qvec vectors (start at kins, end at kouts)
        add_vectors(plotter, [origin] * len(qvecs[:100]), qvecs[-1000:], "red", shaft_scale= 0.05,tip_scale=.1)
        
        plotter.view_xz()
        plotter.background_color = 'white'
        # Add legend and show
        plotter.add_legend([
            ("kins (blue)", "blue"),
            ("kouts (green)", "green"),
            ("qvec (red)", "red"),
        ])

        # Add axes
        plotter.show_axes()  # Displays x, y, z axes in the 3D plot
        plotter.save_graphic(f"{fldr_name}/vectors_map_{time_str}.pdf")
        if show:
            plotter.show()
        plotter.close()
    
    end_time = time.time()
    
    total_time_mins = (end_time - start_time)/60
    if save_parameters:
        
        params = ""
        
        params += f"Simulation lapsed: {total_time_mins} minutes \n\n"
            
        params += f"Unit cell shape: {crystal.unit_cell_shape} degrees\n"
        
        params += f"Unit cell size: {crystal.unit_cell_size} Ang \n"
        
        params += f"Crystal size: {crystal.crystal_size} #unit cells \n "
        
        for key, value in crystal.loc_atoms.items(): 
            
            params += f"Atom: {key} at {value} \n"
            
        if "crystal_orientation" in kwargs:
            
            params += f"Crystal orientation (chi,mu,eta,phi): {crystal.crystal_orientation} \n"
            
        params += "_"* 50
        
        params += "\n\n"
        params += f"X-ray energy: {beam.energy} \n"
        params += f"Lens NA: {beam.focusing_lens_NA} \n"
        
        params += f"Number of convergent vectors: {beam.num_vectors} \n"
        
        params += "_"* 50
        
        params += "\n\n"
        
        params += f"Detector distance: {detector.distance} m \n"
        
        params += f"Detector size: {detector.size} m \n"
        
        params += f"Pixel size: {detector.pixel_size} um \n"
        
        params += f"Selected HKL: {kwargs['hkl']} \n"
        
        params += f"HKLs on detector: {q_indices} \n"
        params += f"HKL range: {range_scale} \n"
        params += f"Elasticity tolerance: {elasticity_tolerance} \n"
        params += f"k-space sampling points: {grid_points} \n"
                
        if "low_pass_filter" in kwargs:
            params += f"Low intensity filter threshold: {lpf} \n"
            #params += f"Low instensity filter fraction set: {kwargs['low_pass_filter']} \n"
            
        
        f = open(f"{fldr_name}/simulation_params_{time_str}.txt", 'w')
        f.write(params)
        f.close()


def simulate_one_bragg_order(crystal, beam, detector, ptycho_scan = False, plot_2d_st = False, plot_3d_st = False, kin_kout_mapping = False, save_parameters = True, n_jobs=8, **kwargs):
    
    """
    Inputs:
        crystal: Crystal Object
        Beam: Beam Object
        Detector: Detector Object
        
        atoms (dict): A dictionary of atoms names and their fractional coordinate in the unit cell.
         
        plot_2d_RL: A boolean value to plot a slice of the 2D samlpes k-space (the shape transform) - Need to then provide a 'slice_index'.
        plot_3d_RL: A boolean value to plot the sampled 3D k-space.
        kin_kout_mapping: A boolean value to plot the kin kout map
        save_parameters: A boolean value to save the parameters of the simulation in a text file. 
        
        shape_normals (np.ndarray): an array representing the normals of the shape of the crystal.
        
        padding (tuple): x,y,z padding amount around the crystal 
        
        crystal_orientation (optional) (tuple): A tuple representing the crystal orientation (chi, mu, eta, phi).
        
        hkl (tuple): The hkl indices of interest. 
        
        mode (string): Determines whether to simulate 1D, 2D or 3D shape transform. 
                
        one_hkl (boolean): whether to simulate one hkl or multiple (works only for streak case)
        
        grid_points (int): The number of points in each dimension in the reciprocal space sampling
        
        range_scale (float): fractional range of k-sapce to sample, from the hkl of interest to the neighbouring hkls
                
        low_pass_filter (float): A fractional value of the maximum intensity below which the diffraction intensities are masked
        
        image_scale (float): if passed, the detector image is scaled as image^(image_scale)   
        
        save_h5_file (string): path to h5 file 
        
        save_folder (string): path to the folder to save simulation results in 
    """
    show = kwargs['show']
    
    if "save_folder" in kwargs:
        path = kwargs["save_folder"]
    
    else:
        path = os.getcwd()
    
    start_time = time.time()
    
    time_str = strftime("%Y-%m-%d_%H.%M")
    fldr_name = f'{path}/simulation_{time_str}'
    
    if not ptycho_scan:
        try:
            os.mkdir(fldr_name)
        except:
            time_str = time_str + "_1"
            pass
        
    crystal.set_real_lattice_vectors()
    
    if 'crystal_orientation' in kwargs:
        crystal.crystal_orientation = kwargs['crystal_orientation']
        crystal.rotate_UC()
        
    crystal.set_recip_lattice_vectors()
    
    shape_normals = kwargs["shape_normals"]
    
    if 'padding' in kwargs:
        padding = kwargs['padding']
    else: 
        padding = (2,2,2)
        
    crystal.set_shape_array(shape_normals, padding=padding)
    
    atoms = kwargs['atoms']
    for key, value in atoms.items():
        crystal.add_atom(key, value)
    
    if ptycho_scan:
        if "illumination_vol" not in kwargs:
            raise ValueError("The illumination volume must be provided if ptycho_scan is True")
        illum_vol = kwargs["illumination_vol"]
        crystal.set_illuminated_volume(illum_vol)
        mask_Ri = is_inside_crystal(crystal.illuminated_indices, crystal.crys_indices)
        
    else: 
        illum_vol = (0,crystal.crystal_size[0],0,crystal.crystal_size[1],0,crystal.crystal_size[2])
        crystal.illuminated_coords = crystal.real_space_coords
        mask_Ri = None
        
    if 'hkl' not in kwargs:
        raise ValueError("A tuple for the hkl indices to simulate must be provided.")
        
    if "grid_points" not in kwargs:
        raise ValueError(" grid points value must be given")
    
    grid_points = kwargs["grid_points"]
    
    if "range_scale" in kwargs:
        range_scale = kwargs['range_scale']
    else: 
        range_scale = .5    
    print("Generating reciprocal lattice vectors.")
    
    crystal.gen_RLS_one_hkl(grid_points = grid_points, hkl= kwargs['hkl'], range_scale = range_scale, ravel=True)
    print("..done.")
    
    print("G vectors calculation")
    
    
    td_shape_transform =False
    one_d_point =False
    full_shape_trans=False
    
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else: 
        mode = '1D'
        
    if mode == '3D':
        full_shape_trans = True
    elif mode == '2D':
        td_shape_transform = True 
    else: 
        one_d_point = True 
    
    
    hkl = np.array([kwargs["hkl"]])
    crystal.gvectors = np.dot(hkl, crystal._recip_space_lattice)
    
    
    if plot_2d_st:
        qvecs_tmp, _ = gen_rlvs_for_one_hkl(hkl=kwargs['hkl'], recip_vecs=crystal._recip_space_lattice, grid_points=grid_points,range_scale=range_scale, ravel=False)
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, qvecs_tmp, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength, mask_Ri ))
        
        if "slice_index" in kwargs:
            slice_index = kwargs["slice_index"]
        else:
            slice_index = grid_points //2
        
        plt.figure(figsize=(8, 8))
        plt.imshow(reciprocal_lattice_intensity[slice_index, :, :])#, extent=(-crystal.q_max, crystal.q_max, -crystal.q_max, crystal.q_max), origin='lower')
        plt.colorbar(label='Intensity')
        plt.title('Reciprocal Lattice (2D Slice)')
        plt.xlabel(r'$q_x$')
        plt.ylabel(r'$q_y$')
        plt.savefig(f"{fldr_name}/2D_RL_slice_{time_str}.jpeg")
        
    if plot_3d_st:
        
        if not plot_2d_st:
            qvecs_tmp,_ = gen_rlvs_for_one_hkl(hkl=kwargs['hkl'], recip_vecs=crystal._recip_space_lattice, grid_points=grid_points,range_scale=range_scale, ravel=False)
        
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, qvecs_tmp, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength, mask_Ri))
            
        Nx,Ny,Nz = reciprocal_lattice_intensity.shape

        grid = pv.wrap(reciprocal_lattice_intensity)

        # Add metadata if needed
        grid.dimensions = (Nx, Ny, Nz)
        grid.spacing = (1, 1, 1)
        
        # Plot
        plotter = pv.Plotter(off_screen=False)
        opacity = [0, 0.6, 0.8, 1]  # Fully transparent below threshold, fully opaque above

        plotter.add_volume(grid, cmap='viridis', opacity=opacity)
        # Plot the volume
        plotter.show_axes()
        plotter.save_graphic(f"{fldr_name}/3D_RL_{time_str}.pdf")
        
        if show:
            plotter.show()
            print("showing")
            
        plotter.close()
        print("3d shape transform plot done")
        
    if "low_pass_filter" in kwargs:
        print("Filtering reciprocal space intensity.")
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, crystal.rlvs, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength, mask_Ri )) 
        lpf = kwargs["low_pass_filter"] * reciprocal_lattice_intensity.max()
        crystal.rlvs = crystal.rlvs[reciprocal_lattice_intensity>lpf]
        print("..done.")
    
    # Main computations of k_out
    print("computing kouts")
    kouts, kin_indices, _ = compute_kout_from_G_kin(crystal.gvectors, beam.kins)
    kins = beam.kins[kin_indices]   
    
    # print("filtering by direction")    
    # max_angle = calc_detector_max_angle(detector.size, detector.distance)
    # kouts, kins = filter_vectors_by_direction(kouts, kins, np.array([0.,0.,1.0]), max_angle)
    
    if "elasticity_tolerance" in kwargs:
        elasticity_tolerance = kwargs["elasticity_tolerance"]
    else: 
        elasticity_tolerance = 1e-6

    print("Filtering by elastic scattering")
    kouts, kins_opt, mask = filter_elastic_scatt(kouts, kins, tolerance=elasticity_tolerance, wavelength= beam.wavelength)    
    qvecs = kouts-kins_opt
    
    print("indexing Q")
    q_indexed = indexQ(qvecs, crystal._recip_space_lattice)
    q_indices = np.unique(q_indexed, axis=0)
    
    nx,ny = (np.array(detector.size)/(np.array(detector.pixel_size)*1e-6)).astype(int)
    full_image = np.zeros((nx,ny))
    
    # Pupil function 
    print("Calculating pupil funciton. ")
    pupil_func = beam.amplitude_profile * np.exp(1j*beam.lens_aberrations)
    kin_pupil_func = pupil_func[mask]
    pupil_image = generate_detector_image(np.abs(pupil_func), kins, detector.size, detector.pixel_size, detector.distance)
    full_image += pupil_image
    
    # A rough way to visualise the kin streak
    # FIX ME
    kin_opt_func = np.ones(kins_opt.shape[0], dtype = complex) 
    kin_streak_image = generate_detector_image(np.abs(kin_opt_func), kins_opt, detector.size, detector.pixel_size, detector.distance)
    full_image = np.where(kin_streak_image != 0, 0, full_image)
    
    
    # 3D scattering ampltiude
    print("Calculating scattering amplitude")
    if full_shape_trans:
        time1 = time.time()
        scat_amp = calculate_scattering_amplitude(crystal.real_space_vectors, crystal.rlvs, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength, mask_Ri,n_jobs=80 )
        intensity = calculate_intensity(scat_amp)
        time2 = time.time()
        print(f"Scattering amplitude done in : {time2-time1}")
        

    else:
        scat_amp_g = calculate_scattering_amplitude(
                crystal.real_space_vectors,
                crystal.gvectors,
                crystal.illuminated_coords,
                crystal.atoms_list,
                list(crystal.loc_atoms.values()),
                beam.wavelength,
                mask_Ri
            )
        
    if td_shape_transform:
        
        crystal.gen_RLS_one_hkl(grid_points = grid_points*4, hkl= kwargs['hkl'], range_scale = range_scale, ravel=True)


    print("Integrating scattering images")
    for k, pf in tqdm(zip(kins_opt, kin_pupil_func)):
        k = np.array([k])
        
        if full_shape_trans:
            kout = crystal.rlvs + k 
            scat_amp_ = pf * scat_amp 
        elif td_shape_transform:
            kout = crystal.rlvs + k 
            kout, _, mask = filter_elastic_scatt(kout, k, tolerance=1e-5, wavelength= beam.wavelength)
            scat_amp_ = pf * calculate_scattering_amplitude(crystal.real_space_vectors, crystal.rlvs[mask], crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength, mask_Ri )
            
        else: 
            kout = crystal.gvectors + k
            scat_amp_ = scat_amp_g * pf
        
        # Detector image 
        image = generate_detector_image(np.abs(scat_amp_), kout, detector.size, detector.pixel_size, detector.distance)
        full_image += image
    
    
        
    plt.figure()
    if "image_scale" in kwargs:
        img_scale = kwargs['image_scale']
    else: 
        img_scale = 1
    
    full_image = full_image**img_scale
    
    validate_qs = kwargs['validate_qs']    
    if validate_qs:
        # Validation
        for i, (kin, kout, qvec) in enumerate(zip(kins_opt, kouts, qvecs)):
            computed_qvec = kout - kin
            angle_ks = calc_angle(kin,kout)
            angle_q = qmag2ttheta(np.linalg.norm(qvec), energy_kev = beam.energy)
            
            assert np.allclose(angle_ks, angle_q), f"Mismatch in angles index {i}: {angle_ks} != {angle_q}"
            assert np.allclose(computed_qvec, qvec), f"Mismatch in vectors at index {i}: {computed_qvec} != {qvec}"
            assert np.allclose(np.linalg.norm(kin), np.linalg.norm(kout)), f"Mismatch in magnitudes at index {i}: {np.linalg.norm(kin)} != {np.linalg.norm(kout)}"
            #print(f"Vector {i} validated: |kin| = {np.linalg.norm(kin):.2f}, |kout| = {np.linalg.norm(kout):.2f}, |qvec| = {np.linalg.norm(qvec):.2f}")

    plt.imshow(full_image)
    plt.colorbar(label=f'Intensity ^({img_scale})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scattering on Detector')
    
    if not ptycho_scan:
        plt.savefig(f"{fldr_name}/detector_image_{time_str}.tiff", format="tiff", dpi=600)
    if show:
        plt.show()
    
    print("detector image done")
    if kin_kout_mapping:
        try:
            plotter = pv.Plotter(off_screen=False)
            # Create PyVista PolyData objects
            cloud1 = pv.PolyData(kins_opt)
            cloud2 = pv.PolyData(kouts)

            # Add the point array
            plotter.add_mesh(cloud1, color="red", point_size=5, render_points_as_spheres=True)
            plotter.add_mesh(cloud2, color="blue", point_size=5, render_points_as_spheres=True)
            # #plotter.add_mesh(cloud3, color="black", point_size=5, render_points_as_spheres=True )

            plotter.view_xy()
            plotter.background_color = 'white'

            # Add legend and show
            plotter.add_legend([
                ("kins (red)", "red"),
                ("kouts (blue)", "blue"),
            ])

            # Add axes to the plot
            plotter.show_axes()
            plotter.save_graphic(f"{fldr_name}/kin_kout_map_{time_str}.pdf")
            if show:
                plotter.show()
            plotter.close()
        except:
            pass 
    
    vector_map = kwargs['vector_map']
    if vector_map: 
        
        #Function to add vectors to the plot

        # Create a PyVista plotter
        plotter = pv.Plotter(off_screen=False)

        def add_vectors(plotter, start_points, vectors, color, shaft_scale = 0.25, tip_scale=.2):
            
            magnitude = np.linalg.norm(vectors[0])
            
            shaft_radius = shaft_scale * 1.0 / magnitude 
            tip_radius = shaft_radius+tip_scale
            
            for start, vec in zip(start_points, vectors):
                arrow = pv.Arrow(start=start, direction=vec, scale='auto', shaft_radius=shaft_radius, tip_radius=tip_radius)
                plotter.add_mesh(arrow, color=color)


        # Origin for kins and kouts
        origin = np.array([0, 0, 0])

        # Add kins vectors
        add_vectors(plotter, [origin] * len(kins_opt[:100]), kins[-1000:], "blue", shaft_scale= 0.1, tip_scale=.02)

        # Add kouts vectors
        add_vectors(plotter, [origin] * len(kouts[:100]), kouts[-1000:], "green", shaft_scale= 0.1,tip_scale=.02)

        # Add qvec vectors (start at kins, end at kouts)
        add_vectors(plotter, [origin] * len(qvecs[:100]), qvecs[-1000:], "red", shaft_scale= 0.05,tip_scale=.1)
        
        plotter.view_xz()
        plotter.background_color = 'white'
        # Add legend and show
        plotter.add_legend([
            ("kins (blue)", "blue"),
            ("kouts (green)", "green"),
            ("qvec (red)", "red"),
        ])

        # Add axes
        plotter.show_axes()  # Displays x, y, z axes in the 3D plot
        plotter.save_graphic(f"{fldr_name}/vectors_map_{time_str}.pdf")
        if show:
            plotter.show()
        plotter.close()
    
    end_time = time.time()
    
    total_time_mins = (end_time - start_time)/60
    
    # Prepare metadata
    metadata = {
        "hkl": kwargs["hkl"],
        "unit_cell_shape": crystal.unit_cell_shape,
        "unit_cell_size" : crystal.unit_cell_size,
        "illum_vol" : illum_vol,
        "crystal_orientation": crystal.crystal_orientation if "crystal_orientation" in kwargs else None,
        "beam_energy": beam.energy,
        "detector_distance": detector.distance,
        "detector_size": detector.size,
        "pixel_size": detector.pixel_size,
        "lens_NA": beam.focusing_lens_NA,
        "grid_points": grid_points,
        "range_scale": range_scale
    }
    metadata = {**metadata, **beam.aberration_coefficients, **beam.amplitude_coefficients}
    
    save_h5_file = kwargs['save_h5_file']
    
    scan_index = ((illum_vol[0]+illum_vol[1])//2, (illum_vol[2]+illum_vol[3])//2)
    save_to_hdf5(save_h5_file, scan_index, full_image, metadata)
    
    if save_parameters:
        
        params = ""
        
        
        params += f"Simulation lapsed: {total_time_mins} minutes \n\n"
            
        params += f"Unit cell shape: {crystal.unit_cell_shape} degrees\n"
        
        params += f"Unit cell size: {crystal.unit_cell_size} Ang \n"
        
        params += f"Crystal size: {crystal.crystal_size} #unit cells \n "
        
        for key, value in crystal.loc_atoms.items(): 
            
            params += f"Atom: {key} at {value} \n"
            
        if "crystal_orientation" in kwargs:
            
            params += f"Crystal orientation (chi,mu,eta,phi): {crystal.crystal_orientation} \n"
        
        
        params += "-"*50
        
        params += f"Ptycho scan = {ptycho_scan} \n"
        if ptycho_scan:
            params += f"illumination volume is {illum_vol} \n"
            
        params += "-"*50 
        
        params += "\n\n"
        params += f"X-ray energy: {beam.energy} \n"
        params += f"Lens NA: {beam.focusing_lens_NA} \n"
        
        params += f"Number of convergent vectors: {beam.num_vectors} \n"
        
        params += "_"* 50
        
        params += "\n\n"
        
        params += f"Detector distance: {detector.distance} m \n"
        
        params += f"Detector size: {detector.size} m \n"
        
        params += f"Pixel size: {detector.pixel_size} um \n"
        
        params += f"Selected HKL: {kwargs['hkl']} \n"

        params += "-"*50 

        
        params += f"HKLs on detector: {q_indices} \n"
        params += f"HKL range: {range_scale} \n"
        params += f"Elasticity tolerance: {elasticity_tolerance} \n"
        params += f"k-space sampling points: {grid_points} \n"
        
        if "low_pass_filter" in kwargs:
            params += f"Low intensity filter threshold: {lpf} \n"
            #params += f"Low instensity filter fraction set: {kwargs["low_pass_filter"]} \n"
            
        f = open(f"{fldr_name}/simulation_params_{time_str}.txt", 'w')
        f.write(params)
        f.close()

def simulate_diff(crystal, beam, detector, plot_2d_RL = False, plot_3d_RL = False, kin_kout_mapping = False, save_parameters = True, **kwargs):
    
    """
    Inputs:
        crystal: Crystal Object
        Beam: Beam Object
        Detector: Detector Object
        
        atoms (dict): A dictionary of atoms names and their fractional coordinate in the unit cell.
         
        plot_2d_RL: A boolean value to plot a slice of the 2D reciprocal lattice - Need to then provide a 'slice_index'.
        plot_3d_RL: A boolean value to plot the sampled 3D reciprocal lattice.
        kin_kout_mapping: A boolean value to plot the kin kout map
        save_parameters: A boolean value to save the parameters of the simulation in a text file. 
        
        shape_normals (np.ndarray): an array representing the normals of the shape of the crystal.
        
        crystal_orientation (optional) (tuple): A tuple representing the crystal orientation (chi, mu, eta, phi).
        
        max_hkl (int): The maximum range of the hkl space to sample. 
        
        grid_points (int): The number of points in each dimension in the reciprocal space sampling
        
        zero_freq_threshold (float): A fractional value for masking out the zero frequency order in reciprocal space sample
        
        low_pass_filter (float): A value for below which the diffraction intensities are masked
        
        image_scale (float): if passed, the detector image is scaled as image^(image_scale)   
        
        save_folder (string): path to the folder to save simulation results in 
    """
    
    if "save_folder" in kwargs:
        path = kwargs["save_folder"]
    
    else:
        path = os.getcwd()
    
    start_time = time.time()
    
    time_str = strftime("%Y-%m-%d_%H.%M")
    fldr_name = f'{path}/simulation_{time_str}'
    os.mkdir(fldr_name)
    
    crystal.set_real_lattice_vectors()
    
    if 'crystal_orientation' in kwargs:
        crystal.crystal_orientation = kwargs['crystal_orientation']
        crystal.rotate_UC()
        
    crystal.set_recip_lattice_vectors()
    
    shape_normals = kwargs["shape_normals"]
    crystal.set_shape_array(shape_normals)
    
    atoms = kwargs['atoms']
    for key, value in atoms.items():
        crystal.add_atom(key, value)
        
    crystal.max_hkl = 3
    
    if "grid_points" not in kwargs:
        raise ValueError(" grid points value must be given")
    
    grid_points = kwargs["grid_points"]
    
    if "zero_freq_threshold" in kwargs:
        thrshld = kwargs["zero_freq_threshold"]
        
    else: 
        thrshld = 0.05
    
    crystal.gen_RLS(grid_points=grid_points, ravel=True, threshold = thrshld)


    if plot_2d_RL:
        data = gen_RLS_from_maxhkl_maskOrigin(crystal.max_hkl, grid_points, *crystal._recip_space_lattice, thrshld, ravel = False)
        # reciprocal_lattice = calculate_form_factor(crys.real_space_vectors, data, crys.real_space_coords )
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, data, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength ))
        
        if "slice_index" in kwargs:
            slice_index = kwargs["slice_index"]
        else:
            slice_index = grid_points //2
        
        plt.figure(figsize=(8, 8))
        plt.imshow(reciprocal_lattice_intensity[slice_index, :, :], extent=(-crystal.q_max, crystal.q_max, -crystal.q_max, crystal.q_max), origin='lower')
        plt.colorbar(label='Intensity')
        plt.title('Reciprocal Lattice (2D Slice)')
        plt.xlabel(r'$q_x$')
        plt.ylabel(r'$q_y$')
        plt.savefig(f"{fldr_name}/2D_RL_slice_{time_str}.jpeg")
    
    if plot_3d_RL:
        
        if not plot_2d_RL: 
            data = gen_RLS_from_maxhkl_maskOrigin(crystal.max_hkl, grid_points, *crystal._recip_space_lattice, 0.05, ravel = False)
            # reciprocal_lattice = calculate_form_factor(crys.real_space_vectors, data, crys.real_space_coords )
            reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, data, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength ))
            
        Nx,Ny,Nz = reciprocal_lattice_intensity.shape

        grid = pv.wrap(reciprocal_lattice_intensity)

        # Add metadata if needed
        grid.dimensions = (Nx, Ny, Nz)
        grid.spacing = (1, 1, 1)
        
        # Plot
        plotter = pv.Plotter()
        opacity = [0, 0, 0.8, 1]  # Fully transparent below threshold, fully opaque above

        plotter.add_volume(grid, cmap='viridis', opacity=opacity)
        # Plot the volume
        plotter.show()
        plotter.screenshot(f"{fldr_name}/3D_RL_{time_str}.jpeg")
    
    if "low_pass_filter" in kwargs:
        
        lpf = kwargs["low_pass_filter"]
        
        reciprocal_lattice_intensity = np.abs(calculate_scattering_amplitude(crystal.real_space_vectors, crystal.rlvs, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength )) ** 2

        crystal.rlvs = crystal.rlvs[reciprocal_lattice_intensity>lpf]
        
    # Main computations of k_out
    kouts, kin_indices, _ = compute_kout_from_G_kin(crystal.rlvs, beam.kins)
    kins = beam.kins[kin_indices]   
        
    max_angle = calc_detector_max_angle(detector.size, detector.distance)
    kouts, kins = filter_vectors_by_direction(kouts, kins, np.array([0.,0.,1.0]), max_angle)
    
    if "elasticity_tolerance" in kwargs:
        elasticity_tolerance = kwargs["elasticity_tolerance"]
    else: 
        elasticity_tolerance = 1e-5
        
    kouts, kins = filter_elastic_scatt(kouts, kins, tolerance=elasticity_tolerance, wavelength=beam.wavelength)
    qvecs = kouts-kins
    qvecs = indexQ(qvecs, crystal._recip_space_lattice)
    q_indices = np.unique(qvecs, axis=0)
    scat_amp = calculate_scattering_amplitude(crystal.real_space_vectors, qvecs, crystal.illuminated_coords, crystal.atoms_list, list(crystal.loc_atoms.values()), beam.wavelength )
    intensity = calculate_intensity(scat_amp)
    
    # Detector image 
    image = generate_detector_image(intensity, kouts, detector.size, detector.pixel_size, detector.distance)

    plt.figure()
    if "image_scale" in kwargs:
        img_scale = kwargs['image_scale']
    else: 
        img_scale = 1
        
    image = image**img_scale
    plt.imshow(image)
    plt.colorbar(label=f'Intensity ^({img_scale})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scattering on Detector')
    plt.show()
    plt.savefig(f"{fldr_name}/detector_image_{time_str}.jpeg")
    
    if kin_kout_mapping:
        # Create PyVista PolyData objects
        cloud1 = pv.PolyData(kins)
        cloud2 = pv.PolyData(kouts)


        plotter = pv.Plotter()

        # Add the point array
        plotter.add_mesh(cloud1, color="red", point_size=5, render_points_as_spheres=True)
        plotter.add_mesh(cloud2, color="blue", point_size=5, render_points_as_spheres=True)
        # #plotter.add_mesh(cloud3, color="black", point_size=5, render_points_as_spheres=True )

        plotter.view_xy()
        plotter.background_color = 'white'

        # Add legend and show
        plotter.add_legend([
            ("kins (red)", "red"),
            ("kouts (blue)", "blue"),
        ])

        # Add axes to the plot
        plotter.show_axes()
        plotter.show()
        plotter.screenshot(f"{fldr_name}/kin_kout_map_{time_str}.jpeg")
    
    end_time = time.time()
    
    total_time_mins = (end_time - start_time)/60
    if save_parameters:
        
        params = ""
        
        params += f"Simulation lapsed: {total_time_mins} minutes \n\n"
            
        params += f"Unit cell shape: {crystal.unit_cell_shape} degrees\n"
        
        params += f"Unit cell size: {crystal.unit_cell_size} Ang \n"
        
        params += f"Crystal size: {crystal.crystal_size} #unit cells \n "
        
        for key, value in crystal.loc_atoms.items(): 
            
            params += f"Atom: {key} at {value} \n"
            
        if "crystal_orientation" in kwargs:
            
            params += f"Crystal orientation (chi,mu,eta,phi): {crystal.crystal_orientation} \n"
            
        params += "_"* 50
        
        params += "\n\n"
        params += f"X-ray energy: {beam.energy} \n"
        params += f"Lens NA: {beam.focusing_lens_NA} \n"
        
        params += f"Number of convergent vectors: {beam.num_vectors} \n"
        
        params += "_"* 50
        
        params += "\n\n"
        
        params += f"Detector distance: {detector.distance} m \n"
        
        params += f"Detector size: {detector.size} m \n"
        
        params += f"Pixel size: {detector.pixel_size} um \n"
        
        params += f"Max HKL: {crystal.max_hkl} \n"
        params += f"Calculated HKLs on detector: {q_indices} \n"
        params += f"k-space sampling points: {grid_points} \n"
        
        params += f"Threshold masking zero frequency: {thrshld} \n"
        
        if "low_pass_filter" in kwargs:
            params += f"Low intensity filter threshold: {lpf} \n"
            
        f = open(f"{fldr_name}/simulation_params_{time_str}.txt", 'w')
        f.write(params)
        f.close()