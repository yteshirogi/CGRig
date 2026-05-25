/* ----------------------------------------------------------------------
   LAMMPS - https://www.lammps.org/
------------------------------------------------------------------------- */

#include "fix_bd_rigid.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "math_extra.h"
#include "memory.h"
#include "random_park.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <unordered_map>

#define EIGEN_DONT_PARALLELIZE
#include <fstream>
#include <regex>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace hydropro_io {

  using Matrix66d = Eigen::Matrix<double,6,6>;
  using Matrix33d = Eigen::Matrix<double,3,3>;
  using Vector6d  = Eigen::Matrix<double,6,1>;
  using Vector3d  = Eigen::Vector3d;

  static inline bool is_number_line(const std::string& s) {
      if (s.size() < 6) return false;
      for (char c: s) {
          if (!(std::isdigit((unsigned char)c) || c==' ' || c=='+' || c=='-' || c=='.' || c=='E' || c=='e'))
              return false;
      }
      return true;
  }

  // read Mobility matrix from HYDROPRO output and convert Å/fs
  static std::pair<Matrix66d, Vector3d> read_hydropro(const std::string& filename) {
      std::ifstream fin(filename);
      if (!fin) throw std::runtime_error("cannot open file: " + filename);
      std::vector<std::string> lines; lines.reserve(4096);
      for (std::string line; std::getline(fin, line); ) lines.push_back(line);

      // Center of diffusion (x)
      std::regex cod_re(R"(Center of diffusion \(x\):\s*([\-\d.Ee\+]+)\s*cm)");
      double cod_cm[3] = {0,0,0};
      int cod_count = 0;
      std::smatch m;
      for (const auto& ln : lines){
          if (std::regex_search(ln, m, cod_re)) {
              if (cod_count < 3) cod_cm[cod_count] = std::stod(m[1]);
              cod_count++;
          }
      }
      if (cod_count < 3)
          throw std::runtime_error("Center of diffusion (x,y,z) not all found");
      Vector3d cod_a(cod_cm[0]*1e8, cod_cm[1]*1e8, cod_cm[2]*1e8); // cm -> Å

      // Mobility matrix
      int start_idx = -1;
      for (int i=0; i<(int)lines.size(); ++i) {
          if (lines[i].find("Generalized (6x6) diffusion matrix") != std::string::npos) {
              start_idx = i; break;
          }
      }
      if (start_idx < 0) throw std::runtime_error("Generalized (6x6) diffusion matrix not found");

      std::vector<std::string> mlines; mlines.reserve(6);
      auto ltrim=[&](std::string& t){ t.erase(t.begin(), std::find_if(t.begin(), t.end(), [](int ch){return !std::isspace(ch);})); };
      auto rtrim=[&](std::string& t){ t.erase(std::find_if(t.rbegin(), t.rend(), [](int ch){return !std::isspace(ch);} ).base(), t.end()); };
      for (int i=start_idx+1; i<(int)lines.size() && (int)mlines.size()<6; ++i) {
          std::string s = lines[i]; ltrim(s); rtrim(s);
          if (is_number_line(s)) mlines.push_back(s);
      }
      if ((int)mlines.size() < 6) throw std::runtime_error("6 lines of diffusion matrix not found");

      Matrix66d D = Matrix66d::Zero();
      for (int r=0; r<6; ++r) {
          std::istringstream iss(mlines[r]);
          for (int c=0; c<6; ++c) {
              double v;
              if (!(iss >> v)) throw std::runtime_error("failed to parse 6x6 diffusion matrix (row="+std::to_string(r)+")");
              D(r,c) = v;
          }
      }
      // unit conversion：tt: cm^2/s → Å^2/fs (=10), rr: s^-1 → fs^-1 (=1e-15), tr/rt: cm/s → Å/fs (=1e-7)
      D.block<3,3>(0,0) *= 10.0;    // translational
      D.block<3,3>(3,3) *= 1e-15;   // rotational
      D.block<3,3>(0,3) *= 1e-7;    // trans-rot
      D.block<3,3>(3,0) *= 1e-7;    // rot-trans

      return {D, cod_a};
  }

  // build friction matrix from mobility matrix
  hydropro_io::BuildZResult read_build_Z(const std::string& filename, double kT,
                                         const Eigen::Vector3d& ref_point_A,
                                         double jitter)
  {
      auto [D_cod, cod_a] = read_hydropro(filename);

      // conversion due to diffusion center and COM
      Vector3d s = ref_point_A - cod_a;
      Matrix33d S;
      S <<     0, -s[2],  s[1],
             s[2],     0, -s[0],
            -s[1],  s[0],     0;
      Matrix66d T = Matrix66d::Identity();
      T.block<3,3>(0,3) = -S;
      Matrix66d Tinv = Matrix66d::Identity();
      Tinv.block<3,3>(3,0) = S;

      Matrix66d D_com = T * (D_cod * Tinv);

      Matrix66d D = 0.5*(D_com + D_com.transpose()); // symmetry
      if (jitter > 0) D.diagonal().array() += jitter;

      Eigen::LLT<Matrix66d> lltD(D);
      if (lltD.info() != Eigen::Success) {
          Matrix66d D2 = D; D2.diagonal().array() += (jitter>0? jitter : 1e-24);
          lltD.compute(D2);
          if (lltD.info() != Eigen::Success) throw std::runtime_error("LLT(D) failed; D not SPD even after jitter");
          D = D2;
      }
      Matrix66d Minv = lltD.solve(Matrix66d::Identity());
      Matrix66d Z = kT * Minv;

      Eigen::LLT<Matrix66d> lltZ(Z);
      if (lltZ.info() != Eigen::Success) {
          Matrix66d Z2 = Z; Z2.diagonal().array() += (jitter>0? jitter : 1e-24);
          lltZ.compute(Z2);
          if (lltZ.info() != Eigen::Success) throw std::runtime_error("LLT(Z) failed; Z not SPD even after jitter");
          Z = Z2;
      }

      BuildZResult out;
      out.D_afs = D;
      out.Z     = Z;
      out.L     = lltZ.matrixL();
      out.CoD_a = cod_a;
      return out;
  }

} // namespace hydropro_io

static std::unordered_map<int, Eigen::Vector3d>
read_comfile_simple(const std::string& path)
{
  std::unordered_map<int, Eigen::Vector3d> m;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open COM file: "+path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    int mid; std::string chain; double x,y,z;
    if (!(iss >> mid >> chain >> x >> y >> z)) continue;
    m[mid] = Eigen::Vector3d(x,y,z);
  }
  return m;
}

static inline unsigned int make_park_seed(unsigned int base, int body_index){
  uint64_t v = (uint64_t)base ^ (0x9E3779B97F4A7C15ull * (uint64_t)(body_index+1));
  v += 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  v = v ^ (v >> 31);
  return 1u + (unsigned int)(v % 899999999ull);
}


using namespace LAMMPS_NS;
using namespace FixConst;

// Quaternion utils
void FixBDRigid::quat_identity(double q[4]) { q[0]=1.0; q[1]=q[2]=q[3]=0.0; }

void FixBDRigid::quat_normalize(double q[4]) {
  double n = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (n > 0) { double inv=1.0/n; q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv; }
  else quat_identity(q);
}

void FixBDRigid::quat_mul(const double a[4], const double b[4], double out[4]) {
  out[0] = a[0]*b[0] - (a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);
  out[1] = a[0]*b[1] + b[0]*a[1] + (a[2]*b[3]-a[3]*b[2]);
  out[2] = a[0]*b[2] + b[0]*a[2] + (a[3]*b[1]-a[1]*b[3]);
  out[3] = a[0]*b[3] + b[0]*a[3] + (a[1]*b[2]-a[2]*b[1]);
}

void FixBDRigid::rotmat_from_quat(const double q[4], double R[3][3]) {
  const double w=q[0], x=q[1], y=q[2], z=q[3];
  const double xx=2*x*x, yy=2*y*y, zz=2*z*z;
  const double xy=2*x*y, xz=2*x*z, yz=2*y*z;
  const double wx=2*w*x, wy=2*w*y, wz=2*w*z;
  R[0][0]=1-yy-zz; R[0][1]=xy-wz;  R[0][2]=xz+wy;
  R[1][0]=xy+wz;   R[1][1]=1-xx-zz;R[1][2]=yz-wx;
  R[2][0]=xz-wy;   R[2][1]=yz+wx;  R[2][2]=1-xx-yy;
}

void FixBDRigid::rotate_vec(const double R[3][3], const double v[3], double out[3]) {
  out[0] = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2];
  out[1] = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2];
  out[2] = R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2];
}

// helpers
static inline bool is_number_token(const char* s) {
  if (!s || !*s) return false;
  char* end=nullptr;
  errno=0;
  const double v = std::strtod(s,&end);
  (void)v;
  if (errno!=0) return false;
  while (end && *end) {
    if (!std::isspace((unsigned char)*end)) return false;
    ++end;
  }
  return true;
}

void FixBDRigid::init_rng_streams()
{
  const int B = (int)bodies_.size();
  rng_streams_.assign(B, nullptr);
  for (int b=0; b<B; ++b) {
    // set determinative seed for each ribid body
    unsigned int sb = make_park_seed(seed_, b);
    rng_streams_[b] = new RanPark(lmp, sb);
  }
}

// ================== ctor/dtor ==================
FixBDRigid::FixBDRigid(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  // usage:
  // fix ID group-ID bd/rigid hydropro MODE <hyd...> kT seed [jitter] [verbose]
  // <hyd...>:
  //   - 単一ファイル: hyd.out
  //   - B個ファイル: hyd1 hyd2 ... hydB   （molid昇順に対応）
  //   - 明示マップ  : molid:file molid:file ...

  // declare 3 column per atom data
  peratom_flag = 1;
  size_peratom_cols = 4; // r0x,r0y,r0z, body_index

  if (narg < 8)
    error->all(FLERR,"Illegal fix bd/rigid command\n"
                     "usage: fix ID group-ID bd/rigid hydropro MODE <hyd...> kT seed [jitter] [verbose]");

  if (std::string(arg[3]) != "hydropro")
    error->all(FLERR,"fix bd/rigid: third keyword must be 'hydropro'");

  mode_str_ = std::string(arg[4]);

  int idx = 5;

  while (idx < narg && !is_number_token(arg[idx])) {
    std::string tok = arg[idx];
    if (tok == "comfile") {
      if (idx+1 > narg) error->all(FLERR, "fix bd/rigid: comfile PATH needs a path");
      comfile_path_ = std::string(arg[idx+1]);
      idx += 2;
      continue;
    }
    hydro_tokens_.emplace_back(tok);
    ++idx;
  }

  if (idx+1 >= narg)
    error->all(FLERR,"fix bd/rigid: missing kT and/or seed");

  kT_     = utils::numeric(FLERR,arg[idx],false,lmp);   // kcal/mol
  seed_   = (unsigned int) utils::inumeric(FLERR,arg[idx+1],false,lmp);
  jitter_ = (idx+2 < narg) ? utils::numeric(FLERR,arg[idx+2],false,lmp) : 0.0;
  verbose_= (idx+3 < narg) ? utils::inumeric(FLERR,arg[idx+3],false,lmp) : 0;

  if      (mode_str_=="full")      use_matrixZ_ = true;
  else if (mode_str_=="diagonal" || mode_str_=="ellipsoid" || mode_str_=="sphere")
    use_matrixZ_ = false;
  else
    error->all(FLERR,"fix bd/rigid: MODE must be one of {full, diagonal, ellipsoid, sphere}");

  atom->add_callback(0); // grow
  atom->add_callback(1); // copy
  atom->add_callback(2); // exchange
}

FixBDRigid::~FixBDRigid()
{
  for (auto *p : rng_streams_) delete p;
  atom->delete_callback(id, 2); // exchange
  atom->delete_callback(id, 1); // copy
  atom->delete_callback(id, 0); // grow
}

// ================== hooks ==================
int FixBDRigid::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

void FixBDRigid::init()
{
  if (!atom->molecule_flag)
    error->all(FLERR,"fix bd/rigid requires atom_style with molecule IDs");

  build_bodies_global();
  if (!comfile_path_.empty()) {
    try {
      com_override_ = read_comfile_simple(comfile_path_);
      if (verbose_ && comm->me==0)
        utils::logmesg(lmp, "FixBDRigid: loaded CoMs from %s (%zu entries)\n",
                       comfile_path_.c_str(), com_override_.size());
    } catch (const std::exception& e) {
      error->all(FLERR, ("COM file read failed: "+std::string(e.what())).c_str());
    }
  }
  compute_initial_r0();
  init_rng_streams();

  // hydropro input
  std::unordered_map<int,std::string> map_molid_file;
  bool any_mapping = false;
  for (const auto& tok : hydro_tokens_) {
    auto pos = tok.find(':');
    if (pos != std::string::npos) {
      any_mapping = true;
      std::string left = tok.substr(0,pos);
      std::string right= tok.substr(pos+1);
      if (left.empty() || right.empty())
        error->all(FLERR,"fix bd/rigid: malformed 'molid:file' token");
      char* end=nullptr;
      long id = std::strtol(left.c_str(), &end, 10);
      if (end==left.c_str() || *end!='\0')
        error->all(FLERR,"fix bd/rigid: non-integer molid in 'molid:file'");
      map_molid_file[(int)id] = right;
    }
  }
  if (any_mapping) {
    if ((int)map_molid_file.size() != (int)bodies_.size())
      error->all(FLERR,"fix bd/rigid: number of 'molid:file' must equal number of molecules in group");
  } else {
    if (hydro_tokens_.size()!=1 && hydro_tokens_.size()!=(size_t)bodies_.size())
      error->all(FLERR,"fix bd/rigid: specify one HYDROPRO file or exactly B files");
  }


  struct ZPack { hydropro_io::Matrix66d Z; };
  std::unordered_map<std::string,ZPack> cache;

  auto load_Z_for_file = [&](const std::string& path)->const ZPack& {
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    hydropro_io::BuildZResult zres;
    try {
      zres = hydropro_io::read_build_Z(path, kT_, Eigen::Vector3d::Zero(), jitter_);
    } catch (const std::exception& e) {
      error->all(FLERR,("HYDROPRO read failed: "+path+" : "+std::string(e.what())).c_str());
    }
    cache[path] = ZPack{zres.Z};
    return cache[path];
  };

  // assign property to each body
  const int B = (int)bodies_.size();
  for (int b=0;b<B;b++) {
    const int mid = bodies_[b].molid;
    std::string file;

    if (any_mapping) {
      auto it = map_molid_file.find(mid);
      if (it==map_molid_file.end())
        error->all(FLERR,"fix bd/rigid: missing file for some molid");
      file = it->second;
    } else if (hydro_tokens_.size()==1) {
      file = hydro_tokens_[0];
    } else {
      file = hydro_tokens_[b];
    }

    const auto& pack = load_Z_for_file(file);
    Eigen::Matrix<double,6,6> Z = pack.Z;

    if (use_matrixZ_) {
      // full: Z = L L^T
      Eigen::LLT<Eigen::Matrix<double,6,6>> llt(Z);
      if (llt.info()!=Eigen::Success) {
        Z.diagonal().array() += (jitter_>0 ? jitter_ : 1e-24);
        llt.compute(Z);
        if (llt.info()!=Eigen::Success) error->all(FLERR,"Cholesky(Z) failed after jitter");
      }
      bodies_[b].Z = Z;
      bodies_[b].L = llt.matrixL();
      bodies_[b].Zdiag.setZero(); bodies_[b].invZdiag.setZero(); bodies_[b].sqrtZdiag.setZero();
    } else {
      // diagonal/ellipsoid/sphere
      Eigen::Matrix<double,6,1> diagZ = Z.diagonal();

      if (mode_str_=="sphere") {
        const double tmean = diagZ.head<3>().mean();
        const double rmean = diagZ.tail<3>().mean();
        diagZ.head<3>().setConstant(tmean);
        diagZ.tail<3>().setConstant(rmean);
      } else if (mode_str_=="ellipsoid") {
        // prolate
        Eigen::Vector3d tr = diagZ.head<3>();
        int idx_long_tr=0; double tr_long = tr.maxCoeff(&idx_long_tr);
        double tr_short = 0.0; for (int i=0;i<3;i++) if (i!=idx_long_tr) tr_short += tr(i);
        tr_short /= 2.0;

        Eigen::Vector3d ro = diagZ.tail<3>();
        int idx_long_ro=0; double ro_long = ro.maxCoeff(&idx_long_ro);
        double ro_short = 0.0; for (int i=0;i<3;i++) if (i!=idx_long_ro) ro_short += ro(i);
        ro_short /= 2.0;

        for (int i=0;i<3;i++) {
          diagZ(i)   = (i==idx_long_tr)? tr_long : tr_short;
          diagZ(i+3) = (i==idx_long_ro)? ro_long : ro_short;
        }
      }
      Eigen::Matrix<double,6,1> invZ, sqrtZ;
      for (int i=0;i<6;i++) {
        const double z = (diagZ[i] > 1e-30 ? diagZ[i] : 1e-30);
        invZ[i]  = 1.0 / z;
        sqrtZ[i] = std::sqrt(z);
      }
      bodies_[b].Z.setZero(); bodies_[b].L.setZero();
      bodies_[b].Zdiag = diagZ;
      bodies_[b].invZdiag = invZ;
      bodies_[b].sqrtZdiag = sqrtZ;
    }

    if (verbose_ && comm->me==0) {
      utils::logmesg(lmp, "FixBDRigid map: mol %d <- %s\n", mid, file.c_str());
    }
  }

  if (verbose_ && comm->me==0) {
    utils::logmesg(lmp, "FixBDRigid: nbodies = %d (mode=%s)\n", (int)bodies_.size(), mode_str_.c_str());
  }
}

void FixBDRigid::setup(int vflag) { (void)vflag; }

// rigid body build
void FixBDRigid::build_bodies_global()
{
  bodies_.clear();
  mol2body_.clear();

  const int nlocal = atom->nlocal;
  tagint *molid = atom->molecule;
  int *mask = atom->mask;

  // gather unique molid in group
  std::vector<int> loc; loc.reserve(nlocal);
  for (int i=0;i<nlocal;i++) if (mask[i] & groupbit) loc.push_back(molid[i]);
  std::sort(loc.begin(), loc.end());
  loc.erase(std::unique(loc.begin(), loc.end()), loc.end());

  // gather size
  int lsz = (int)loc.size();
  std::vector<int> allsz(comm->nprocs);
  MPI_Allgather(&lsz, 1, MPI_INT, allsz.data(), 1, MPI_INT, world);

  // gather data
  int tot = 0; std::vector<int> disp(comm->nprocs, 0);
  for (int p=0;p<comm->nprocs;++p) {
    disp[p] = tot;
    tot += allsz[p];
  }
  std::vector<int> buf(tot);
  MPI_Allgatherv(loc.data(), lsz, MPI_INT, buf.data(), allsz.data(), disp.data(), MPI_INT, world);

  // unique in global
  std::sort(buf.begin(), buf.end());
  buf.erase(std::unique(buf.begin(), buf.end()), buf.end());

  bodies_.resize(buf.size());
  for (size_t b=0;b<buf.size();++b) {
    auto &B = bodies_[b];
    B.molid = buf[b];
    quat_identity(B.q);
    B.com[0]=B.com[1]=B.com[2]=0.0;
    B.F[0]=B.F[1]=B.F[2]=0.0;
    B.T[0]=B.T[1]=B.T[2]=0.0;
    mol2body_[B.molid] = (int)b;
  }
}

void FixBDRigid::compute_initial_r0()
{
  static constexpr double CA_MASS = 12.0107;
  static constexpr double CA_RADIUS = 1.7;
  const double Iref = (2.0/5.0) * CA_MASS * (CA_RADIUS*CA_RADIUS);
  const Eigen::Matrix3d IrefMat = Iref * Eigen::Matrix3d::Identity();

  // allcate per atom data
  grow_arrays(atom->nmax);
  double **x = atom->x;
  tagint *molid = atom->molecule;
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  const int Bn = (int)bodies_.size();

  // cast anker atom coordinate
  for (int b=0;b<Bn;b++) {
    const int mid = bodies_[b].molid;
    // extract anker atom
    long long local_min_tag = LLONG_MAX;
    double ax=0, ay=0, az=0;
    for (int i=0;i<atom->nlocal;i++) {
      if (!(mask[i] & groupbit)) continue;
      if (molid[i] != mid) continue;
      if ((long long)tag[i] < local_min_tag) {
        local_min_tag = (long long)tag[i];
        ax = x[i][0]; ay = x[i][1]; az = x[i][2];
      }
    }
    // global min tag
    long long global_min_tag;
    MPI_Allreduce(&local_min_tag, &global_min_tag, 1, MPI_LONG_LONG, MPI_MIN, world);

    // detect anker rank
    int have = (local_min_tag == global_min_tag) ? 1 : 0;
    int owner_rank = have ? comm->me : -1;
    MPI_Allreduce(MPI_IN_PLACE, &owner_rank, 1, MPI_INT, MPI_MAX, world);
    // broadcast anker position
    if (comm->me == owner_rank) {
      // keep ax,ay,az
    } else {
      ax=ay=az=0.0;
    }
    MPI_Bcast(&ax, 1, MPI_DOUBLE, owner_rank, world);
    MPI_Bcast(&ay, 1, MPI_DOUBLE, owner_rank, world);
    MPI_Bcast(&az, 1, MPI_DOUBLE, owner_rank, world);
    const Eigen::Vector3d A(ax, ay, az);

    // calculate COM
    Eigen::Vector3d COM(0,0,0);
    if (com_override_.count(mid)) {
      // align COM to anker image
      Eigen::Vector3d cext = com_override_[mid];
      double dx = cext[0] - ax;
      double dy = cext[1] - ay;
      double dz = cext[2] - az;
      domain->minimum_image(FLERR, dx, dy, dz);
      COM = A + Eigen::Vector3d(dx, dy, dz);
    } else {
      // allreduce
      double sumx = 0, sumy = 0, sumz = 0;
      long long cnt = 0;
      for (int i=0;i<atom->nlocal;i++) {
        if (!(mask[i] & groupbit)) continue;
        if (molid[i] != mid) continue;
        double dx = x[i][0] - ax;
        double dy = x[i][1] - ay;
        double dz = x[i][2] - az;
        domain->minimum_image(FLERR, dx, dy, dz);
        sumx += (ax + dx); sumy += (ay + dy); sumz += (az + dz);
        cnt++;
      }
      double gsum[4] = {sumx,sumy,sumz,(double)cnt};
      MPI_Allreduce(MPI_IN_PLACE, gsum, 4, MPI_DOUBLE, MPI_SUM, world);
      if (gsum[3] > 0) {
        COM = Eigen::Vector3d(gsum[0], gsum[1], gsum[2]) / gsum[3];
      } else {
        COM = A; // fallback
      }
    }
    bodies_[b].com[0] = COM[0]; bodies_[b].com[1] = COM[1]; bodies_[b].com[2] = COM[2];

    // inertia tensor
    Eigen::Matrix3d Iloc = Eigen::Matrix3d::Zero();
    for (int i=0;i<atom->nlocal;i++) {
      if (!(mask[i] & groupbit)) continue;
      if (molid[i] != mid) continue;
      double dx = x[i][0] - ax;
      double dy = x[i][1] - ay;
      double dz = x[i][2] - az;
      domain->minimum_image(FLERR, dx, dy, dz);
      Eigen::Vector3d xu(ax+dx, ay+dy, az+dz);
      Eigen::Vector3d r = xu - COM;
      const double r2 = r.squaredNorm();
      Iloc += IrefMat + CA_MASS * (r2 * Eigen::Matrix3d::Identity() - r * r.transpose());
    }
    // allreduce 3x3
    double buf[9] = {
      Iloc(0,0), Iloc(0,1), Iloc(0,2),
      Iloc(1,0), Iloc(1,1), Iloc(1,2),
      Iloc(2,0), Iloc(2,1), Iloc(2,2)
    };
    MPI_Allreduce(MPI_IN_PLACE, buf, 9, MPI_DOUBLE, MPI_SUM, world);
    Eigen::Matrix3d I;
    I << buf[0], buf[1], buf[2],
         buf[3], buf[4], buf[5],
         buf[6], buf[7], buf[8];
    // principal axis
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(I);
    Eigen::Matrix3d R = (es.info() == Eigen::Success) ? es.eigenvectors() : Eigen::Matrix3d::Identity();
    if (R.determinant() < 0.0) R.col(2) *= -1.0; // right-handed
    Eigen::Quaterniond Q0(R);
    bodies_[b].q[0] = Q0.w(); bodies_[b].q[1]=Q0.x(); bodies_[b].q[2]=Q0.y(); bodies_[b].q[3]=Q0.z();
    quat_normalize(bodies_[b].q);

    // set local r0
    const Eigen::Matrix3d RT = R.transpose();
    for (int i=0;i<atom->nlocal;i++) {
      if (!(mask[i] & groupbit)) continue;
      if (molid[i] != mid) continue;
      double dx = x[i][0] - ax;
      double dy = x[i][1] - ay;
      double dz = x[i][2] - az;
      domain->minimum_image(FLERR, dx, dy, dz);
      Eigen::Vector3d xu(ax+dx, ay+dy, az+dz);
      Eigen::Vector3d rel = RT * (xu - COM);
      r0a[i][0] = rel[0]; r0a[i][1] = rel[1]; r0a[i][2] = rel[2];
      bindexa[i] = b;
    }
  }
}

// For MPI
void FixBDRigid::build_owner_layout_(int Bn)
{
  // cache
  int world_size = 1, world_rank = 0;
  MPI_Comm_size(world, &world_size);
  MPI_Comm_rank(world, &world_rank);
  nprocs_ = world_size; me_ = world_rank;
  if (Bn_cached_ == Bn && (int)rank_offsets_.size() == nprocs_ + 1) return;

  owner_.resize(Bn);
  rank_counts_.assign(nprocs_, 0);
  for (int b=0; b<Bn; ++b) {
    int r = b % nprocs_;
    owner_[b] = r;
    rank_counts_[r] += 1;
  }
  rank_offsets_.assign(nprocs_ + 1, 0);
  for (int r=0; r<nprocs_; ++r) rank_offsets_[r + 1] = rank_offsets_[r] + rank_counts_[r];

  rank_body_ids_.resize(Bn);
  std::vector<int> cursor = rank_counts_;
  for (int r=0; r<nprocs_; ++r) cursor[r] = 0;
  for (int b=0; b<Bn; ++b) {
    int r = owner_[b];
    int pos = rank_offsets_[r] + cursor[r]++;
    rank_body_ids_[pos] = b;
  }
  // my bodies view
  my_bodies_.clear();
  if (rank_counts_[me_] > 0) {
    my_bodies_.insert(my_bodies_.end(),
      rank_body_ids_.begin() + rank_offsets_[me_],
      rank_body_ids_.begin() + rank_offsets_[me_] + rank_counts_[me_]);
  }
  // Allgatherv
  const int S = 7; // (com, q)
  recvcounts_dbl_.resize(nprocs_);
  displs_dbl_.resize(nprocs_);
  for (int r=0; r<nprocs_; ++r) recvcounts_dbl_[r] = rank_counts_[r] * S;
  displs_dbl_[0] = 0;
  for (int r=0; r<nprocs_-1; ++r) displs_dbl_[r + 1] = displs_dbl_[r] + recvcounts_dbl_[r];

  // state buffers
  send_state_.assign(rank_counts_[me_] * S, 0.0);
  all_state_.assign(Bn * S, 0.0);
  Bn_cached_ = Bn;
}

// ================== force/torque ==================
void FixBDRigid::post_force(int vflag)
{
  // static double t_final_integrate_sum = 0.0;
  // static long n_final_integrate = 0;

  // double t0 = MPI_Wtime();

  (void)vflag;
  double **f = atom->f;

  const int Bn = (int)bodies_.size();

  if (Bn==0) return;
  // prepare rotation matrix for each bodies
  if ((int)Rm_.size() != 9*Bn) Rm_.resize(9*Bn);
  // std::vector<double> Rm(9*Bn);
  for (int b=0;b<Bn;++b) {
    double R[3][3]; rotmat_from_quat(bodies_[b].q, R);
    for (int r=0; r<3; r++) for (int c=0; c<3; c++) Rm_[9*b+3*r+c] = R[r][c];
  }

  // static double t_FT_sum = 0.0;
  // static long n_FT = 0;

  // double t0_FT = MPI_Wtime();

  std::vector<double> FTg(6*Bn, 0.0); // thread local [0..3B):F, [3B..6B):T
  // // vector reduction
  double *FT = FTg.data();
  const double *RM = Rm_.data();
  // accumulate forces/torques
  #pragma omp parallel for schedule(static) reduction(+:FT[:6*Bn])
  for (int i=0; i<atom->nlocal; ++i) {
    const int b = bindexa[i];
    if (b < 0 || b >= Bn) continue;

    const double fx = f[i][0], fy = f[i][1], fz = f[i][2];
    FT[3*b+0] += fx; FT[3*b+1] += fy; FT[3*b+2] += fz;
    const double *R = &RM[9*b];
    const double rx = R[0]*r0a[i][0] + R[1]*r0a[i][1] + R[2]*r0a[i][2];
    const double ry = R[3]*r0a[i][0] + R[4]*r0a[i][1] + R[5]*r0a[i][2];
    const double rz = R[6]*r0a[i][0] + R[7]*r0a[i][1] + R[8]*r0a[i][2];

    // torque
    FT[3*Bn+3*b+0] += ry*fz - rz*fy;
    FT[3*Bn+3*b+1] += rz*fx - rx*fz;
    FT[3*Bn+3*b+2] += rx*fy - ry*fx;
  }

  // double t1_FT = MPI_Wtime();
  // t_FT_sum += (t1_FT - t0_FT);
  // n_FT++;
  // if (update->ntimestep % 2000 == 0 && comm->me == 0) {
  //   double avg = (n_FT > 0) ? (1e3 * t_FT_sum / double(n_FT)) : 0.0;
  //   printf("[bd/rigid] post_force>FT loop avg = %.6f ms/step\n", avg);
  //   t_FT_sum = 0.0;
  //   n_FT = 0;
  // }


  // Allreduce
  MPI_Allreduce(MPI_IN_PLACE, FTg.data(), 6*Bn, MPI_DOUBLE, MPI_SUM, world);
  for (int b=0; b<Bn; b++) {
    bodies_[b].F[0] = FTg[3*b+0]; bodies_[b].F[1] = FTg[3*b+1]; bodies_[b].F[2] = FTg[3*b+2];
    bodies_[b].T[0] = FTg[3*Bn+3*b+0]; bodies_[b].T[1] = FTg[3*Bn+3*b+1]; bodies_[b].T[2] = FTg[3*Bn+3*b+2];
  }

  // double t1 = MPI_Wtime();
  // t_final_integrate_sum += (t1 - t0);
  // n_final_integrate++;
  // if (update->ntimestep % 2000 == 0 && comm->me == 0) {
  //   double avg = (n_final_integrate > 0) ? (1e3 * t_final_integrate_sum / double(n_final_integrate)) : 0.0;
  //   printf("[bd/rigid] post_force avg = %.6f ms/step\n", avg);
  //   t_final_integrate_sum = 0.0;
  //   n_final_integrate = 0;
  // }

}

// ================== integrate ==================
void FixBDRigid::final_integrate()
{
  // static double t_final_integrate_sum = 0.0;
  // static long n_final_integrate = 0;

  // double t0 = MPI_Wtime();

  const double dt = update->dt;


  // owner-rank
  const int Bn = (int)bodies_.size();
  build_owner_layout_(Bn);

  // update owner body
  #pragma omp parallel for schedule(static)
  for (int j=0; j<(int)my_bodies_.size(); ++j) {
    const int b = my_bodies_[j];
    bd_step_one_body_(dt, b);
  }


  // Allreduce
  Q_.assign(Bn*4, 0.0);
  COM_.assign(Bn*3, 0.0);
  for (int b : my_bodies_) {
    const int baseQ = 4 * b;
    const int baseC = 3 * b;
    // normalize
    const double q0 = bodies_[b].q[0], q1 = bodies_[b].q[1], q2 = bodies_[b].q[2], q3 = bodies_[b].q[3];
    const double invn = 1.0 / sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    Q_[baseQ+0] = q0*invn;
    Q_[baseQ+1] = q1*invn;
    Q_[baseQ+2] = q2*invn;
    Q_[baseQ+3] = q3*invn;
    COM_[baseC+0] = bodies_[b].com[0];
    COM_[baseC+1] = bodies_[b].com[1];
    COM_[baseC+2] = bodies_[b].com[2];
  }
  MPI_Allreduce(MPI_IN_PLACE, Q_.data(), Bn*4, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, COM_.data(), Bn*3, MPI_DOUBLE, MPI_SUM, world);
  for (int b=0; b<Bn; ++b) {
    const int baseQ = 4 * b;
    const int baseC = 3 * b;
    bodies_[b].q[0] = Q_[baseQ+0];
    bodies_[b].q[1] = Q_[baseQ+1];
    bodies_[b].q[2] = Q_[baseQ+2];
    bodies_[b].q[3] = Q_[baseQ+3];
    bodies_[b].com[0] = COM_[baseC+0];
    bodies_[b].com[1] = COM_[baseC+1];
    bodies_[b].com[2] = COM_[baseC+2];
  }

  // reconstruct R from new q
  #pragma omp parallel for schedule(static)
  for (int b=0; b<Bn; ++b) {
    double *R = &Rm_[9*b];
    const double q0 = bodies_[b].q[0], q1 = bodies_[b].q[1], q2 = bodies_[b].q[2], q3 = bodies_[b].q[3];
    const double a = q0, b1 = q1, c = q2, d = q3;
    const double aa=a*a, bb=b1*b1, cc=c*c, dd=d*d;
    const double ab=a*b1, ac=a*c, ad=a*d, bc=b1*c, bd=b1*d, cd=c*d;
    // right-handed
    R[0] = aa+bb-cc-dd; R[1] = 2*(bc-ad);   R[2] = 2*(bd+ac);
    R[3] = 2*(bc+ad);   R[4] = aa-bb+cc-dd; R[5] = 2*(cd-ab);
    R[6] = 2*(bd-ac);   R[7] = 2*(cd+ab);   R[8] = aa-bb-cc+dd;
  }
  // reconstruct atom coords in each rank
  double **x = atom->x;
  imageint *image = atom->image;
  const int nlocal = atom->nlocal;
  #pragma omp parallel for schedule(static)
  for (int i=0; i<nlocal; ++i) {
    const int b = bindexa[i];
    if (b<0 || b>=Bn) continue;
    const double *R = &Rm_[9*b];
    const double rx = R[0]*r0a[i][0] + R[1]*r0a[i][1] + R[2]*r0a[i][2];
    const double ry = R[3]*r0a[i][0] + R[4]*r0a[i][1] + R[5]*r0a[i][2];
    const double rz = R[6]*r0a[i][0] + R[7]*r0a[i][1] + R[8]*r0a[i][2];
    x[i][0] = bodies_[b].com[0] + rx;
    x[i][1] = bodies_[b].com[1] + ry;
    x[i][2] = bodies_[b].com[2] + rz;
    domain->remap(x[i], image[i]);
  }


}

void FixBDRigid::wrap_com_into_box(Body& B) const
{
  double xc[3] = {B.com[0], B.com[1], B.com[2]};
  imageint img = 0;
  domain->remap(xc, img);
  B.com[0]=xc[0]; B.com[1]=xc[1]; B.com[2]=xc[2];
}

void FixBDRigid::bd_step(double dt)
{
  const double s_noise = std::sqrt(std::max(0.0, 2.0*kT_*dt));

  const int Bn = (int)bodies_.size();
  #pragma omp parallel for schedule(static)
  for (int b=0; b<Bn; b++) {
    auto &B = bodies_[b];
    // noise
    Eigen::Matrix<double,6,1> xi;
    RanPark *rp = rng_streams_[b];
    xi[0] = rp->gaussian();
    xi[1] = rp->gaussian();
    xi[2] = rp->gaussian();
    xi[3] = rp->gaussian();
    xi[4] = rp->gaussian();
    xi[5] = rp->gaussian();

    // lab→body
    double Rm[3][3]; rotmat_from_quat(B.q, Rm);
    double RT[3][3] = {
      {Rm[0][0], Rm[1][0], Rm[2][0]},
      {Rm[0][1], Rm[1][1], Rm[2][1]},
      {Rm[0][2], Rm[1][2], Rm[2][2]}
    };
    double Fb[3] = {
      RT[0][0]*B.F[0] + RT[0][1]*B.F[1] + RT[0][2]*B.F[2],
      RT[1][0]*B.F[0] + RT[1][1]*B.F[1] + RT[1][2]*B.F[2],
      RT[2][0]*B.F[0] + RT[2][1]*B.F[1] + RT[2][2]*B.F[2]
    };
    double Tb[3] = {
      RT[0][0]*B.T[0] + RT[0][1]*B.T[1] + RT[0][2]*B.T[2],
      RT[1][0]*B.T[0] + RT[1][1]*B.T[1] + RT[1][2]*B.T[2],
      RT[2][0]*B.T[0] + RT[2][1]*B.T[1] + RT[2][2]*B.T[2]
    };

    Eigen::Matrix<double,6,1> rhs;
    rhs << dt*Fb[0], dt*Fb[1], dt*Fb[2], dt*Tb[0], dt*Tb[1], dt*Tb[2];

    if (use_matrixZ_) {
      rhs.noalias() += s_noise * (B.L * xi);

      Eigen::Matrix<double,6,1> y  = B.L.template triangularView<Eigen::Lower>().solve(rhs);
      Eigen::Matrix<double,6,1> dy = B.L.transpose().template triangularView<Eigen::Upper>().solve(y);

      const double dRb[3] = { dy[0], dy[1], dy[2] };
      const double dRlab[3] = {
        Rm[0][0]*dRb[0] + Rm[0][1]*dRb[1] + Rm[0][2]*dRb[2],
        Rm[1][0]*dRb[0] + Rm[1][1]*dRb[1] + Rm[1][2]*dRb[2],
        Rm[2][0]*dRb[0] + Rm[2][1]*dRb[1] + Rm[2][2]*dRb[2]
      };
      B.com[0] += dRlab[0]; B.com[1] += dRlab[1]; B.com[2] += dRlab[2];

      const double dthx = dy[3], dthy = dy[4], dthz = dy[5];
      const double ang2 = dthx*dthx + dthy*dthy + dthz*dthz;
      double dq[4];
      if (ang2 > 0) {
        const double ang  = std::sqrt(ang2);
        const double half = 0.5*ang;
        const double s    = std::sin(half) / ang;
        dq[0] = std::cos(half);
        dq[1] = s*dthx; dq[2] = s*dthy; dq[3] = s*dthz;
      } else { dq[0]=1.0; dq[1]=dq[2]=dq[3]=0.0; }
      double qnew[4];
      quat_mul(B.q, dq, qnew);
      std::memcpy(B.q, qnew, sizeof(qnew));
      quat_normalize(B.q);
    } else {
      for (int k=0;k<6;k++) rhs[k] += s_noise * B.sqrtZdiag[k] * xi[k];
      Eigen::Matrix<double,6,1> dy;
      for (int k=0;k<6;k++) dy[k] = B.invZdiag[k] * rhs[k];

      const double dRb[3] = { dy[0], dy[1], dy[2] };
      const double dRlab[3] = {
        Rm[0][0]*dRb[0] + Rm[0][1]*dRb[1] + Rm[0][2]*dRb[2],
        Rm[1][0]*dRb[0] + Rm[1][1]*dRb[1] + Rm[1][2]*dRb[2],
        Rm[2][0]*dRb[0] + Rm[2][1]*dRb[1] + Rm[2][2]*dRb[2]
      };
      B.com[0] += dRlab[0]; B.com[1] += dRlab[1]; B.com[2] += dRlab[2];

      const double dthx = dy[3], dthy = dy[4], dthz = dy[5];
      const double ang2 = dthx*dthx + dthy*dthy + dthz*dthz;
      double dq[4];
      if (ang2 > 0) {
        const double ang  = std::sqrt(ang2);
        const double half = 0.5*ang;
        const double s    = std::sin(half) / ang;
        dq[0] = std::cos(half);
        dq[1] = s*dthx; dq[2] = s*dthy; dq[3] = s*dthz;
      } else { dq[0]=1.0; dq[1]=dq[2]=dq[3]=0.0; }
      double qnew[4];
      quat_mul(B.q, dq, qnew);
      std::memcpy(B.q, qnew, sizeof(qnew));
      quat_normalize(B.q);
    }
  }
}

void FixBDRigid::bd_step_one_body_(double dt, int b)
{
  const double s_noise = std::sqrt(std::max(0.0, 2.0*kT_*dt));

  // noise
  Eigen::Matrix<double,6,1> xi;
  RanPark *rp = rng_streams_[b];
  xi[0] = rp->gaussian();
  xi[1] = rp->gaussian();
  xi[2] = rp->gaussian();
  xi[3] = rp->gaussian();
  xi[4] = rp->gaussian();
  xi[5] = rp->gaussian();

  auto &B = bodies_[b];
  // lab→body
  double Rm[3][3]; rotmat_from_quat(B.q, Rm);
  double RT[3][3] = {
    {Rm[0][0], Rm[1][0], Rm[2][0]},
    {Rm[0][1], Rm[1][1], Rm[2][1]},
    {Rm[0][2], Rm[1][2], Rm[2][2]}
  };
  double Fb[3] = {
    RT[0][0]*B.F[0] + RT[0][1]*B.F[1] + RT[0][2]*B.F[2],
    RT[1][0]*B.F[0] + RT[1][1]*B.F[1] + RT[1][2]*B.F[2],
    RT[2][0]*B.F[0] + RT[2][1]*B.F[1] + RT[2][2]*B.F[2]
  };
  double Tb[3] = {
    RT[0][0]*B.T[0] + RT[0][1]*B.T[1] + RT[0][2]*B.T[2],
    RT[1][0]*B.T[0] + RT[1][1]*B.T[1] + RT[1][2]*B.T[2],
    RT[2][0]*B.T[0] + RT[2][1]*B.T[1] + RT[2][2]*B.T[2]
  };

  Eigen::Matrix<double,6,1> rhs;
  rhs << dt*Fb[0], dt*Fb[1], dt*Fb[2], dt*Tb[0], dt*Tb[1], dt*Tb[2];

  if (use_matrixZ_) {
    rhs.noalias() += s_noise * (B.L * xi);

    Eigen::Matrix<double,6,1> y  = B.L.template triangularView<Eigen::Lower>().solve(rhs);
    Eigen::Matrix<double,6,1> dy = B.L.transpose().template triangularView<Eigen::Upper>().solve(y);

    const double dRb[3] = { dy[0], dy[1], dy[2] };
    const double dRlab[3] = {
      Rm[0][0]*dRb[0] + Rm[0][1]*dRb[1] + Rm[0][2]*dRb[2],
      Rm[1][0]*dRb[0] + Rm[1][1]*dRb[1] + Rm[1][2]*dRb[2],
      Rm[2][0]*dRb[0] + Rm[2][1]*dRb[1] + Rm[2][2]*dRb[2]
    };
    B.com[0] += dRlab[0]; B.com[1] += dRlab[1]; B.com[2] += dRlab[2];

    const double dthx = dy[3], dthy = dy[4], dthz = dy[5];
    const double ang2 = dthx*dthx + dthy*dthy + dthz*dthz;
    double dq[4];
    if (ang2 > 0) {
      const double ang  = std::sqrt(ang2);
      const double half = 0.5*ang;
      const double s    = std::sin(half) / ang;
      dq[0] = std::cos(half);
      dq[1] = s*dthx; dq[2] = s*dthy; dq[3] = s*dthz;
    } else { dq[0]=1.0; dq[1]=dq[2]=dq[3]=0.0; }
    double qnew[4];
    quat_mul(B.q, dq, qnew);
    std::memcpy(B.q, qnew, sizeof(qnew));
  } else {
    for (int k=0;k<6;k++) rhs[k] += s_noise * B.sqrtZdiag[k] * xi[k];
    Eigen::Matrix<double,6,1> dy;
    for (int k=0;k<6;k++) dy[k] = B.invZdiag[k] * rhs[k];

    const double dRb[3] = { dy[0], dy[1], dy[2] };
    const double dRlab[3] = {
      Rm[0][0]*dRb[0] + Rm[0][1]*dRb[1] + Rm[0][2]*dRb[2],
      Rm[1][0]*dRb[0] + Rm[1][1]*dRb[1] + Rm[1][2]*dRb[2],
      Rm[2][0]*dRb[0] + Rm[2][1]*dRb[1] + Rm[2][2]*dRb[2]
    };
    B.com[0] += dRlab[0]; B.com[1] += dRlab[1]; B.com[2] += dRlab[2];

    const double dthx = dy[3], dthy = dy[4], dthz = dy[5];
    const double ang2 = dthx*dthx + dthy*dthy + dthz*dthz;
    double dq[4];
    if (ang2 > 0) {
      const double ang  = std::sqrt(ang2);
      const double half = 0.5*ang;
      const double s    = std::sin(half) / ang;
      dq[0] = std::cos(half);
      dq[1] = s*dthx; dq[2] = s*dthy; dq[3] = s*dthz;
    } else { dq[0]=1.0; dq[1]=dq[2]=dq[3]=0.0; }
    double qnew[4];
    quat_mul(B.q, dq, qnew);
    std::memcpy(B.q, qnew, sizeof(qnew));
  }
}


double FixBDRigid::compute_scalar()
{
  return (double)bodies_.size();
}

void FixBDRigid::grow_arrays(int nmax) {
  int old = nmax_alloc_;
  memory->grow(r0a, nmax, 3, "bd/rigid:r0a");
  memory->grow(bindexa, nmax,   "bd/rigid:bidx");
  // zero initialize
  for (int i=old; i<nmax; ++i) {
    r0a[i][0] = r0a[i][1] = r0a[i][2] = 0.0;
    bindexa[i] = -1;
  }
  nmax_alloc_ = nmax;
}
void FixBDRigid::copy_arrays(int i, int j, int delflag) {
  (void)delflag;
  r0a[j][0] = r0a[i][0];
  r0a[j][1] = r0a[i][1];
  r0a[j][2] = r0a[i][2];
  bindexa[j] = bindexa[i];
}
int FixBDRigid::pack_exchange(int i, double *buf) {
  buf[0] = r0a[i][0];
  buf[1] = r0a[i][1];
  buf[2] = r0a[i][2];
  buf[3] = (double)bindexa[i];
  return 4;
}
int FixBDRigid::unpack_exchange(int nlocal, double *buf) {
  r0a[nlocal][0] = buf[0];
  r0a[nlocal][1] = buf[1];
  r0a[nlocal][2] = buf[2];
  bindexa[nlocal] = (int)llround(buf[3]);
  return 4;
}
