# CGRig
This is the LAMMPS plugin used in below:
> CGRig: A Rigid-Body Protein Model with Residue-Level Interaction Sites for Long-Time and Large-Scale Protein Assembly Simulation. Yosuke Teshirogi, Tohru Terada. doi:10.64898/2026.03.21.713350, 2026.

## Installation
1. Put the files in `src/` into the `YOUR_LAMMPS_SRC_PATH/src/EXTRA-FIX/`.
2. Build/compile the LAMMPS, enabling `EXTRA-FIX` and explicitly specifying the Eigen path

Then, you can use our simulation method as `fix bd/rigid`

## Performance tips

If you want a better performance, you can convert `gauss/cut` interaction (native contact term) to `table` interaction because `gauss/cut` is not implemented in the KOKKOS acceleration package but `table` is implemented in KOKKOS manner.


