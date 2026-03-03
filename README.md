# pycbd_3d
> Atomistic simulation of 3D reciprocal-space diffraction volumes from convergent beam diffraction (CBD) of crystals in the Bragg geometry.

A Python package for atomistic modelling of **Convergent Beam Diffraction (CBD)** crystallography experiments.

It enables the **full 3D reciprocal space** — building up a volumetric diffraction dataset by sampling across orientations, tilts, or rocking angles. This mirrors experimental techniques such as **3D-CBD**, **precession electron diffraction (PED)**, and **orientation-resolved diffraction tomography**, where a series of diffraction patterns are collected as the crystal is systematically rotated.

It enables researchers to:

- Generate complete 3D reciprocal-space maps of a crystal's diffraction signal
- Simulate tilt-series and rotation-series CBD datasets for comparison with experiment
- Explore how 3D diffraction volumes encode strain, defects, and crystal morphology
- Produce ground-truth 3D datasets for algorithm development and machine learning

## Background: From 2D Patterns to 3D Diffraction Volumes

A single 2D CBD pattern captures the Ewald sphere slice through reciprocal space at one orientation. By rotating or tilting the crystal through a series of angles, the Ewald sphere sweeps through reciprocal space, and the full 3D diffraction intensity distribution $I(\mathbf{q})$ can be reconstructed.

`simulate_3d` models this process directly — computing the atomistic structure factor across a 3D grid of reciprocal lattice points and assembling the resulting volumetric intensity map. This is particularly valuable for:

- **Serial crystallography** — where many randomly oriented crystals each contribute a 2D snapshot that must be merged into a 3D dataset
- **Bragg ptychography / CDI** — where 3D diffraction volumes are inverted to recover real-space structure
- **Diffraction tomography** — systematic tilt-series acquisition and 3D reconstruction

---

## Features

- **3D reciprocal-space volume construction** — assembles a complete $I(h, k, l)$ intensity map across the sampled reciprocal lattice
- **Atomistic structure factor computation** — calculates diffracted intensities from atomic positions, scattering factors, and Debye-Waller factors for each reciprocal lattice point
- **Convergent beam geometry** — correctly accounts for the cone of incident wave vectors at each orientation, spreading intensity into diffraction disks rather than delta-function spots
- **Bragg geometry** — operates in the reflection geometry appropriate for surface and thin-crystal CBD experiments
- **Orientation / tilt-series sampling** — simulate diffraction over a defined range of crystal orientations or rocking angles to build up the 3D volume
- **Ewald sphere sweep** — tracks the Ewald sphere's intersection with reciprocal space across all sampled orientations for physically accurate intensity accumulation
- **3D volume output** — returns a voxelised 3D intensity array in reciprocal space coordinates, ready for visualisation or inversion
- **NumPy / Matplotlib compatible** — outputs standard arrays for seamless integration with scientific Python workflows

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/ahmed2903/pycbd.git
cd pycbd
pip install .
```

**Requirements:** Python 3.7+, NumPy, SciPy, Matplotlib

---

## Usage

```python
from pycbd.simulate_3d import Simulation3D

# Define crystal structure (e.g. Silicon)
sim = Simulation3D(
    lattice_params=(5.43, 5.43, 5.43, 90, 90, 90),  # a, b, c (Å), α, β, γ (°)
    atomic_basis=[
        ("Si", [0.00, 0.00, 0.00]),
        ("Si", [0.25, 0.25, 0.25]),
    ],
    beam_energy_keV=17.0,            # Incident beam energy
    convergence_angle_mrad=5.0,      # Semi-angle of convergence
    detector_distance_mm=100.0,      # Sample-to-detector distance
    zone_axis=[0, 0, 1],             # Central beam direction (h k l)
    tilt_range_deg=(-10, 10),        # Angular range to sweep
    tilt_steps=50,                   # Number of orientation samples
)

# Run the 3D simulation
volume = sim.run()

# Visualise a slice through the 3D reciprocal-space volume
volume.plot_slice(axis='kz', index=0)

# Access the raw 3D intensity array (h, k, l voxels)
intensity_volume = volume.data  # shape: (Nh, Nk, Nl)
```

A worked example is also available in the repository notebook:

```
simulate_nb.ipynb
```

---

## Output

`simulate_3d` returns a 3D diffraction volume object containing:

| Output | Description |
|---|---|
| **3D intensity volume** | Voxelised $I(h, k, l)$ array in reciprocal space |
| **Reflection list** | Miller indices, reciprocal-space coordinates, and integrated intensities for all sampled reflections |
| **Orientation stack** | 2D pattern stack across the sampled tilt/rotation series |
| **Reciprocal-space axes** | Calibrated $q_x$, $q_y$, $q_z$ coordinate arrays for the output volume |


---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/ahmed2903/pycbd).

---

## License

See [LICENSE](LICENSE) for details.

---

## Citation

If `pycbd` contributes to published work, please cite:

```
@software{pycbd,
  author = {ahmed2903},
  title  = {pycbd: Atomistic Simulation of Convergent Beam Diffraction},
  url    = {https://github.com/ahmed2903/pycbd},
}
```
