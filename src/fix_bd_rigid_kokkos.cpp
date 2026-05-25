#include "fix_bd_rigid_kokkos.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <mpi.h>

using namespace LAMMPS_NS;

FixBDRigidKokkos::FixBDRigidKokkos(LAMMPS *lmp, int narg, char **arg)
: FixBDRigid(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = dynamic_cast<AtomKokkos*>(atom);
  // datamask_read   = EMPTY_MASK;
  datamask_read   = F_MASK;
  datamask_modify = X_MASK | IMAGE_MASK | F_MASK;
}

int FixBDRigidKokkos::setmask()
{
    return FixBDRigid::setmask();
}

void FixBDRigidKokkos::post_force(int vflag)
{
  // fallback to CPU
  if (atomKK) atomKK->sync(Host, F_MASK);
  FixBDRigid::post_force(vflag);
  if (atomKK) atomKK->modified(Host, F_MASK);
}

void FixBDRigidKokkos::final_integrate()
{
  // fallback to CPU
  if (atomKK) atomKK->sync(Host, X_MASK | IMAGE_MASK);
  FixBDRigid::final_integrate();
  if (atomKK) atomKK->modified(Host, X_MASK | IMAGE_MASK);
}
