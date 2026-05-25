/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - https://www.lammps.org/
------------------------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(bd/rigid,FixBDRigid);
// clang-format on
#else

#ifndef LMP_FIX_BD_RIGID_H
#define LMP_FIX_BD_RIGID_H

#include "fix.h"
#include <vector>
#include <unordered_map>
#include <string>

#ifdef LAMMPS_KOKKOS
#include "kokkos.h"
#include "atom_kokkos.h"
#endif


#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace hydropro_io {
    using Matrix66d = Eigen::Matrix<double,6,6>;
    using Vector3d  = Eigen::Vector3d;
    struct BuildZResult {
        Matrix66d D_afs;
        Matrix66d Z;
        Matrix66d L;
        Vector3d  CoD_a;
    };
    BuildZResult read_build_Z(const std::string& filename, double kT,
                              const Eigen::Vector3d& ref_point_A = Eigen::Vector3d::Zero(),
                              double jitter = 0.0);
}

static std::unordered_map<int, Eigen::Vector3d> read_comfile_simple(const std::string& path);
static unsigned int make_park_seed(unsigned int base, int body_index);
namespace LAMMPS_NS {

class RanPark;

class FixBDRigid : public Fix {
 public:
  FixBDRigid(class LAMMPS *, int, char **);
  ~FixBDRigid() override;

  int setmask() override;
  void init() override;
  void setup(int) override;

  void post_force(int) override;
  void final_integrate() override;
  double compute_scalar() override;

  // per-atom properties
  void grow_arrays(int nmax) override;
  void copy_arrays(int i, int j, int delflag) override;
  int pack_exchange(int i, double *buf) override;
  int unpack_exchange(int nlocal, double *buf) override;
 protected:
  struct Body {
    int molid;

    double com[3];
    double q[4];
    double F[3], T[3];

    Eigen::Matrix<double,6,6> Z;
    Eigen::Matrix<double,6,6> L;

    Eigen::Matrix<double,6,1> Zdiag;
    Eigen::Matrix<double,6,1> invZdiag;
    Eigen::Matrix<double,6,1> sqrtZdiag;
  };

  // Quaternion helpers
  static void quat_identity(double q[4]);
  static void quat_normalize(double q[4]);
  static void quat_mul(const double a[4], const double b[4], double out[4]);
  static void rotmat_from_quat(const double q[4], double R[3][3]);
  static void rotate_vec(const double R[3][3], const double v[3], double out[3]);

  // PBC
  void wrap_com_into_box(Body& B) const;

  // Rigid build
  void build_bodies_global();
  void compute_initial_r0();
  void refresh_local_indices();

  // Integrator
  void bd_step(double dt);
  void bd_step_one_body_(double dt, int b);

  // Args / params
  std::string mode_str_; // full|diagonal|ellipsoid|sphere
  double kT_;
  unsigned int seed_;
  double jitter_;
  int verbose_;
  bool use_matrixZ_ = true;

  // hydropro
  std::vector<std::string> hydro_tokens_;
  // COM
  std::string comfile_path_;
  std::unordered_map<int, Eigen::Vector3d> com_override_;

  // Internals
  std::vector<Body> bodies_;
  std::unordered_map<int,int> mol2body_;
  RanPark *rng_ = nullptr;
  std::vector<RanPark*> rng_streams_;
  void init_rng_streams();

  // per-atom properties
  double **r0a = nullptr; // per-atom reference position in body frame
  int    *bindexa = nullptr; // nmax
  int nmax_alloc_ = 0;  // previously allocated size

  // owner-rank
  int nprocs_ = 1;
  int me_ = 0;
  int Bn_cached_ = -1;
  std::vector<int> owner_;
  std::vector<int> rank_counts_;
  std::vector<int> rank_offsets_;
  std::vector<int> rank_body_ids_;
  std::vector<int> my_bodies_;
  // Allgatherv related
  std::vector<int> recvcounts_dbl_, displs_dbl_;
  std::vector<double> send_state_;
  std::vector<double> all_state_;
  // helper
  void build_owner_layout_(int Bn);

  // TEST
  std::vector<double> Rm_;
  std::vector<double> Q_; // quaternion
  std::vector<double> COM_; // com
};

} // namespace LAMMPS_NS

#endif
#endif
