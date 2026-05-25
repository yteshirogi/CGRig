/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - https://www.lammps.org/
------------------------------------------------------------------------- */
#ifdef FIX_CLASS
FixStyle(bd/rigid/kk, FixBDRigidKokkos);
#else

#ifndef LMP_FIX_BD_RIGID_KOKKOS_H
#define LMP_FIX_BD_RIGID_KOKKOS_H

#include "fix_bd_rigid.h"
#include "atom_kokkos.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace LAMMPS_NS {

class FixBDRigidKokkos : public FixBDRigid {
    public:
        FixBDRigidKokkos(LAMMPS *lmp, int narg, char **arg);
        ~FixBDRigidKokkos() override = default;

        int setmask() override;
        void post_force(int vflag) override;
        void final_integrate() override;

    private:
        AtomKokkos *atomKK = nullptr;
};

} // namespace LAMMPS_NS

#endif // LMP_FIX_BD_RIGID_KOKKOS_H
#endif // FIX_CLASS
