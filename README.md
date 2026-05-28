# CGRig
This is the LAMMPS plugin used in below:
> CGRig: A Rigid-Body Protein Model with Residue-Level Interaction Sites for Long-Time and Large-Scale Protein Assembly Simulation. Yosuke Teshirogi, Tohru Terada. doi:10.64898/2026.03.21.713350, 2026.

## Installation
1. Put the files in `src/` into the `YOUR_LAMMPS_SRC_PATH/src/EXTRA-FIX/`.
2. Build/compile the LAMMPS, enabling `EXTRA-FIX` and explicitly specifying the Eigen path

For example:
```

cd YOUR_LAMMPS_SRC_PATH

rm -rf build
mkdir build
cd build

cmake ../cmake \
 -DCMAKE_INSTALL_PREFIX=/work/gw43/w43008/apps/lammps/22Jul2025_cgrig_kokkos \
 -DCMAKE_C_COMPILER=gcc \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_Fortran_COMPILER=gfortran \
 -DCMAKE_CXX_STANDARD=17 \
 -DBUILD_MPI=yes \
 -DBUILD_OMP=yes \
 -DPKG_OPENMP=yes \
 -DPKG_MOLECULE=yes \
 -DPKG_EXTRA-DUMP=yes \
 -DPKG_EXTRA-PAIR=yes \
 -DBUILD_SHARED_LIBS=yes \
 -DCMAKE_CXX_FLAGS="-IYOUR_EIGEN_PATH/include/eigen3" \
 -DPKG_KOKKOS=yes \
 -DKokkos_ENABLE_OPENMP=yes \
 -DPKG_EXTRA-FIX=yes
make -j16
make install
```

Then, you can use our simulation method as `fix bd/rigid`.

## How to run

For example, run for 16 tubulin-dimer system using KOKKOS acceleration:
```
lmp \
  -k on g 1 t 72 -sf kk -pk kokkos newton on neigh half \
  -in tub32.in \
  -l tub32.log \
  -var DATA tub32.data \
  -var SEED 12345 \
  -var OUTDCD tub32.dcd \
  -var OUTRESTART restart \
  -var MODE full
```


## Performance tips

If you want a better performance, you can convert `gauss/cut` interaction (native contact term) to `table` interaction because `gauss/cut` is not implemented in the KOKKOS acceleration package but `table` is implemented in KOKKOS manner.


