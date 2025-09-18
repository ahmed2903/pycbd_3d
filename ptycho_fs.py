import numpy as np 

def is_inside_crystal(illuminated_coords, crystal_coords):
        
        # Compute min/max boundaries of the crystal in each axis
        mins = crystal_coords.min(axis=0)  # [xmin, ymin, zmin]
        maxs = crystal_coords.max(axis=0)  # [xmax, ymax, zmax]
        
        # Check if points are inside all three coordinate bounds
        mask = np.all((illuminated_coords >= mins) & (illuminated_coords < maxs), axis=1).astype(np.uint8)  

        return mask

def ptycho_scan_volumes(crystal_size, stride, beam_focus, padding=(0,0,0)):
    
    zi = 0
    zf = int(crystal_size[2])
    
    
    xsize = crystal_size[0] + 2* padding[0]
    ysize = crystal_size[1] + 2* padding[1]
    
    x_stride, y_stride = stride
    x_focus, y_focus = beam_focus
    
    illumination_vols = []
        
    for xs in range(0, xsize-x_focus+1, x_stride):
        
        for ys in range(0,ysize-y_focus+1, y_stride):
            
            xi = xs
            xf = xs + x_focus
            
            yi = ys
            yf = ys + y_focus
            
            illumination_vols.append((int(xi),int(xf),int(yi),int(yf),int(zi),int(zf)))
            
    return illumination_vols