#pragma once

#include <eigen/Eigen/Dense>
#include "../config.h"
#include "../det/det.h"
#include "../solver/sparse_matrix.h"
#include "integrals.h"
#include "point_group.h"

using namespace Eigen;

class RDM {
  typedef std::pair<unsigned, unsigned> index_t;
 public:
  RDM(const Integrals& integrals_,
      const std::vector<Det>& dets_,
      const std::vector<std::vector<double>>& coefs_): integrals(integrals_), 
    dets(dets_),
    coefs(coefs_),
    n_orbs(integrals_.n_orbs),
    n_up(integrals_.n_up),
    n_dn(integrals_.n_dn),
    n_states(coefs.size()),
    time_sym(Config::get<bool>("time_sym", false)) {
  }

  void get_1rdm();
  
  void get_1rdm_unpacked();

  void get_2rdm(
      const std::vector<std::vector<size_t>>& connections);

  void get_2rdm(
      const SparseMatrix& hamiltonian_matrix);

  void get_1rdm_from_2rdm();
  
  // Spin-resolved RDM functions
  void get_spin_1rdm();
  
  void get_spin_2rdm(
      const std::vector<std::vector<size_t>>& connections);
  
  void get_spin_2rdm(
      const SparseMatrix& hamiltonian_matrix);
  
  void dump_spin_1rdm() const;
  
  void dump_spin_2rdm(const bool dump_csv = false) const;
  
  double spin_one_rdm_elem(const unsigned p, const unsigned q, const bool is_up) const;
  
  double spin_two_rdm_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const std::string& spin_config) const;
  
  void dump_1rdm() const;

  void dump_2rdm(const bool dump_csv = false) const;

  double one_rdm_elem(const unsigned, const unsigned) const;

  double two_rdm_elem(const unsigned, const unsigned, const unsigned, const unsigned) const;

  void clear();

 private:
  const Integrals& integrals;

  const std::vector<Det>& dets;
 
  const std::vector<std::vector<double>>& coefs;

  const unsigned n_orbs, n_up, n_dn, n_states;

  const bool time_sym;

  MatrixXd one_rdm;

  std::vector<double> two_rdm;
  
  // Spin-resolved RDMs
  MatrixXd spin_one_rdm_up;    // γ^α(p,q) = ⟨Ψ|a†_{p,α} a_{q,α}|Ψ⟩
  MatrixXd spin_one_rdm_dn;    // γ^β(p,q) = ⟨Ψ|a†_{p,β} a_{q,β}|Ψ⟩
  
  std::vector<double> spin_two_rdm_aa;  // Γ^{αα}(p,q,r,s)
  std::vector<double> spin_two_rdm_bb;  // Γ^{ββ}(p,q,r,s)
  std::vector<double> spin_two_rdm_ab;  // Γ^{αβ}(p,q,r,s)
  std::vector<double> spin_two_rdm_ba;  // Γ^{βα}(p,q,r,s)

  inline size_t combine4_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  int permfac_ccaa(HalfDet halfket, const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;

  void compute_energy_from_rdm() const;

  void get_2rdm_pair(const Det& connected_det, const size_t connected_ind, const Det& this_det, const size_t this_ind);

  void get_2rdm_elements(
      const Det& connected_det,
      const size_t j_det,
      const Det& this_det,
      const size_t i_det,
      const double tr_factor);

  void MPI_Allreduce_2rdm();

  void write_in_1rdm(const unsigned p, const unsigned q, const double factor, const size_t i_det, const size_t j_det);

  void write_in_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const double factor, const size_t i_det, const size_t j_det);
  
  // Spin-resolved helper functions
  void write_in_spin_1rdm(const unsigned p, const unsigned q, const double factor, const size_t i_det, const size_t j_det, const bool is_up);
  
  void write_in_spin_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const double factor, const size_t i_det, const size_t j_det, const std::string& spin_config);
  
  void get_spin_2rdm_pair(const Det& connected_det, const size_t connected_ind, const Det& this_det, const size_t this_ind);
  
  void get_spin_2rdm_elements(
      const Det& connected_det,
      const size_t j_det,
      const Det& this_det,
      const size_t i_det,
      const double tr_factor);
  
  void MPI_Allreduce_spin_2rdm();
  
  inline size_t combine4_spin_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const;
};
