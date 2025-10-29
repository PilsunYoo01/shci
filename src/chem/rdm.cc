#include "rdm.h"

#include <fgpl/src/hash_map.h>
#include <math.h>
#include <stdio.h>
#include <eigen/Eigen/Dense>
#include "../parallel.h"
#include "../timer.h"
#include "../util.h"


void RDM::get_1rdm() {
  //=====================================================
  // Create 1RDM using the variational wavefunction
  //
  // Created: Y. Yao, June 2018
  //=====================================================
  Timer::start("get 1rdm");
  const std::vector<unsigned>& orb_sym = integrals.orb_sym;

  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the index of a det
  std::unordered_map<Det, size_t, DetHasher> det2ind;
  for (size_t i = 0; i < dets.size(); i++) {
    det2ind[dets[i]] = i;
  }

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    const Det& this_det = dets[i_det];

    const std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
    const std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

    // up electrons
    for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
      unsigned p = occ_up[i_elec];
      for (unsigned r = 0; r < n_orbs; r++) {
        if (orb_sym[p] != orb_sym[r]) continue;
        if (p != r && this_det.up.has(r)) continue;

        Det new_det = this_det;
        new_det.up.unset(p);
        new_det.up.set(r);

	size_t connected_ind;
        double factor = 1.;
        if (!time_sym) {
          if (det2ind.count(new_det) == 1) {
	    connected_ind = det2ind[new_det];
          } else
            continue;
	} else {
	  Det new_det_proxy = new_det;
	  if (new_det.up > new_det.dn)
	    new_det_proxy.reverse_spin();
	  if (det2ind.count(new_det_proxy) == 1) {
	    connected_ind = det2ind[new_det_proxy];
	  } else
	    continue;
	  if (this_det.up != this_det.dn) factor *= Util::SQRT2_INV;
	  if (new_det.up != new_det.dn) factor *= Util::SQRT2_INV;
	}
	factor *= this_det.up.diff(new_det.up).permutation_factor; 

        write_in_1rdm(p, r, factor, i_det, connected_ind);
      }  // r
    }  // i_elec

    if (time_sym && this_det.up != this_det.dn) {
      Det this_det_reversed = this_det;
      this_det_reversed.reverse_spin();

      const auto& occ_up_reversed = occ_dn;

      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up_reversed[i_elec];
        for (unsigned r = 0; r < n_orbs; r++) {
          if (orb_sym[p] != orb_sym[r]) continue;
          if (p != r && this_det_reversed.up.has(r)) continue;
  
          Det new_det = this_det_reversed;
          new_det.up.unset(p);
          new_det.up.set(r);
  
	  size_t connected_ind;
          double factor =  Util::SQRT2_INV; // since this_det is time_sym packed
	  Det new_det_proxy = new_det;
  	  if (new_det.up > new_det.dn)
  	    new_det_proxy.reverse_spin();
  	  if (det2ind.count(new_det_proxy) == 1) {
	    connected_ind = det2ind[new_det_proxy];
  	  } else
  	    continue; 
  	  if (new_det.up != new_det.dn) factor *= Util::SQRT2_INV;
          factor *= this_det_reversed.up.diff(new_det.up).permutation_factor;

          write_in_1rdm(p, r, factor, i_det, connected_ind);
        }  // r
      }  // i_elec

    }

    // dn electrons
    if (!time_sym) {  // If time_sym, then the up excitations will equal the dn excitations
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        for (unsigned r = 0; r < n_orbs; r++) {
          if (orb_sym[p] != orb_sym[r]) continue;
          if (p != r && this_det.dn.has(r)) continue;

          Det new_det = this_det;
          new_det.dn.unset(p);
          new_det.dn.set(r);

          if (det2ind.count(new_det) == 0)
            continue;

          write_in_1rdm(p, r, this_det.dn.diff(new_det.dn).permutation_factor, 
			  i_det, det2ind[new_det]);
        }  // r
      }  // i_elec
    }
  }  // i_det

  if (time_sym) { // dn electrons part
    one_rdm *= 2.;
  }
  Timer::end();
}

void RDM::get_1rdm_unpacked() {
  //=====================================================
  // Create 1RDM using the variational wavefunction
  // This version assumes time_sym unpacked dets
  //=====================================================
  const std::vector<unsigned int>& orb_sym = integrals.orb_sym;

  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the index of a det
  std::unordered_map<Det, size_t, DetHasher> det2ind;
  for (size_t i = 0; i < dets.size(); i++) {
    det2ind[dets[i]] = i;
  }

#pragma omp parallel for
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    const Det& this_det = dets[i_det];

    const std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
    const std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

    // up electrons
    for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
      unsigned p = occ_up[i_elec];
      for (unsigned r = 0; r < n_orbs; r++) {
        if (orb_sym[p] != orb_sym[r]) continue;
        if (p != r && this_det.up.has(r)) continue;

        Det new_det = this_det;
        new_det.up.unset(p);
        new_det.up.set(r);

        if (det2ind.count(new_det) == 0)
          continue;
          
        write_in_1rdm(p, r, this_det.up.diff(new_det.up).permutation_factor,
		       i_det, det2ind[new_det]);
      }  // r
    }  // i_elec

    // dn electrons
    if (!time_sym) {  // If time_sym, then the up excitations will equal to the dn excitations
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        for (unsigned r = 0; r < n_orbs; r++) {
          if (orb_sym[p] != orb_sym[r]) continue;
          if (p != r && this_det.dn.has(r)) continue;

          Det new_det = this_det;
          new_det.dn.unset(p);
          new_det.dn.set(r);

          if (det2ind.count(new_det) == 0)
            continue;

          write_in_1rdm(p, r, this_det.dn.diff(new_det.dn).permutation_factor, 
			  i_det, det2ind[new_det]);
        }  // r
      }  // i_elec
    }
  }  // i_det

  if (time_sym) {
    one_rdm *= 2.;
  }
  Timer::checkpoint("computing 1RDM");
}

void RDM::dump_1rdm() const {
  if (Parallel::is_master()) {
    FILE* pFile;
    pFile = fopen("1rdm.csv", "w");
    fprintf(pFile, "p,r,1rdm\n");
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned r = p; r < n_orbs; r++) {
        const double rdm_pr = one_rdm(p, r);
        if (std::abs(rdm_pr) < 1e-9) continue;
        fprintf(
            pFile,
            "%d,%d,%#.15g\n",
            integrals.orb_order[p],
            integrals.orb_order[r],
            rdm_pr);
      }
    }
    fclose(pFile);
  }
}

void RDM::write_in_1rdm(const unsigned p, const unsigned q, const double factor, 
		const size_t i_det, const size_t j_det) {
  // <psi_(i_det) | c^dagger_p c_q | psi_(j_det)>
  double value = 0.;
  for (unsigned i_state=0; i_state<n_states; i_state++) value += coefs[i_state][i_det] * coefs[i_state][j_det];
  value *= factor;
#pragma omp atomic
  one_rdm(p, q) += value;
}

inline size_t RDM::combine4_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  const size_t a = p * n_orbs + s;
  const size_t b = q * n_orbs + r;
  if (a > b) {
    return (a * (a + 1)) / 2 + b;
  } else {
    return (b * (b + 1)) / 2 + a;
  }
}

int RDM::permfac_ccaa(HalfDet halfket, const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  // Calculate the permutation factor of
  // c^dag_p c^dag_q c_r c_s |halfket>

  unsigned counter = 0;

  // annihilation operators
  std::vector<unsigned> orbs_a{s, r};
  for (unsigned iorb = 0; iorb < 2; iorb++) {
    unsigned orb = orbs_a[iorb];
    counter += halfket.bit_till(orb);
    halfket.unset(orb);
  }

  // creation operators
  std::vector<unsigned> orbs_c{q, p};
  for (unsigned iorb = 0; iorb < 2; iorb++) {
    unsigned orb = orbs_c[iorb];
    counter += halfket.bit_till(orb);
    halfket.set(orb);
  }

  if (counter % 2 == 0)
    return 1;
  else
    return -1;
}

void RDM::dump_2rdm(const bool dump_csv) const {
  auto orb_order = integrals.orb_order;
  auto orb_order_inv = integrals.orb_order_inv;

  if (Parallel::is_master()) {
    std::cout << "writing out 2RDM\n";

    if (dump_csv) {  // .csv format
      FILE* pFile = fopen("2rdm.csv", "w");
      fprintf(pFile, "p,q,r,s,2rdm\n");
      for (unsigned p = 0; p < n_orbs; p++) {
        for (unsigned q = p; q < n_orbs; q++) {
          for (unsigned s = 0; s < n_orbs; s++) {
            for (unsigned r = 0; r < n_orbs; r++) {
              if (p == q && s > r) continue;
              const double rdm_pqrs = two_rdm_elem(p, q, r, s);
              if (std::abs(rdm_pqrs) < 1.0e-9) continue;
              fprintf(
                  pFile,
                  "%d,%d,%d,%d,%#.15g\n",
                  orb_order[p],
                  orb_order[q],
                  orb_order[r],
                  orb_order[s],
                  rdm_pqrs);
            }
          }
        }
      }
      fclose(pFile);
    } else {  // .txt format
      FILE* pFile;
      pFile = fopen("spatialRDM.txt", "w");

      fprintf(pFile, "%d\n", n_orbs);

      for (unsigned p = 0; p < n_orbs; p++) {
        for (unsigned q = 0; q < n_orbs; q++) {
          for (unsigned s = 0; s < n_orbs;
               s++) {  // r and s switched in keeping with Dice conventions
            for (unsigned r = 0; r < n_orbs; r++) {
              if (std::abs(
                      two_rdm_elem(
                          orb_order_inv[p], orb_order_inv[q], orb_order_inv[r], orb_order_inv[s])) >
                  1.e-6)
                fprintf(
                    pFile,
                    "%3d   %3d   %3d   %3d   %10.8g\n",
                    p,
                    q,
                    s,
                    r,
                    two_rdm_elem(
                        orb_order_inv[p], orb_order_inv[q], orb_order_inv[r], orb_order_inv[s]));
            }  // r
          }  // s
        }  // q
      }  // p
    }
  }
}

void RDM::get_2rdm(const std::vector<std::vector<size_t>>& connections) {
  //=====================================================
  // Create spatial 2RDM using the variational wavefunction
  // and Hamiltonian connections.
  //
  // D_pqrs corresponds to <a^+_p a^+_q a_r a_s>
  // Four different spin configurations
  // (1) p: up, q: up, r: up, s: up
  // (2) p: dn, q: dn, r: dn, s: dn
  // (3) p: dn, q: up, r: up, s: dn
  // (4) p: up, q: dn, r: dn, s: up
  //
  // Symmetry (p,s)<->(q,r) is used to reduce storage by half.
  //
  // When time_sym is used, the input wavefunction and Hamiltonian
  // connections should NOT be unpacked.
  //
  // Created: Y. Yao, August 2018
  // Modified: Y. Yao, October 2018: MPI compatibility
  //=====================================================
  Timer::start("get 2rdm");
  const size_t n_orbs_ll = n_orbs;
  const size_t size_two_rdm = n_orbs_ll * n_orbs_ll * (n_orbs_ll * n_orbs_ll + 1) / 2;
  two_rdm.resize(size_two_rdm, 0.);

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < connections.size(); i_det++) {
    Det this_det = dets[i_det];

    for (size_t j_det = 0; j_det < connections[i_det].size(); j_det++) {
      size_t connected_ind = connections[i_det][j_det];
      Det connected_det = dets[connected_ind];
      get_2rdm_pair(connected_det, connected_ind, this_det, i_det);
    }
  }
      
  MPI_Allreduce_2rdm();
  Timer::end();
}


void RDM::get_2rdm(const SparseMatrix& hamiltonian_matrix) {
  // This overloaded version takes SparseMatrix as connections type.
  Timer::start("get 2rdm");
  const size_t n_orbs_ll = n_orbs;
  const size_t size_two_rdm = n_orbs_ll * n_orbs_ll * (n_orbs_ll * n_orbs_ll + 1) / 2;
  two_rdm.resize(size_two_rdm, 0.);

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < hamiltonian_matrix.count_n_rows(); i_det++) {
    Det this_det = dets[i_det];
    auto hamiltonian_row = hamiltonian_matrix.get_row(i_det);
    
    for (size_t j_det = 0; j_det < hamiltonian_row.size(); j_det++) {
      size_t connected_ind = hamiltonian_row.get_index(j_det);
      Det connected_det = dets[connected_ind];
      get_2rdm_pair(connected_det, connected_ind, this_det, i_det);
    }
  }
      
  MPI_Allreduce_2rdm();
  Timer::end();
}

inline void RDM::get_2rdm_pair(const Det& connected_det, 
       const size_t connected_ind, const Det& this_det, const size_t this_ind) {
  if (!time_sym)
    get_2rdm_elements(connected_det, connected_ind, this_det, this_ind, 1.);
  else {
    if (this_det.up == this_det.dn) {
      if (connected_det.up == connected_det.dn) {
        get_2rdm_elements(connected_det, connected_ind, this_det, this_ind, 1.);
      } else {
        Det connected_det_rev = connected_det;
        connected_det_rev.reverse_spin();
  
        get_2rdm_elements(connected_det, connected_ind, this_det, this_ind, Util::SQRT2_INV);
        get_2rdm_elements(connected_det_rev, connected_ind, this_det, this_ind, Util::SQRT2_INV);
      }
    } else {
      if (connected_det.up == connected_det.dn) {
        Det this_det_rev = this_det;
        this_det_rev.reverse_spin();
  
        get_2rdm_elements(connected_det, connected_ind, this_det, this_ind, Util::SQRT2_INV);
        get_2rdm_elements(connected_det, connected_ind, this_det_rev, this_ind, Util::SQRT2_INV);
      } else {
        Det connected_det_rev = connected_det;
        connected_det_rev.reverse_spin();
        Det this_det_rev = this_det;
        this_det_rev.reverse_spin();
  
        get_2rdm_elements(connected_det, connected_ind, this_det, this_ind, 0.5);
        if (this_ind != connected_ind)
          get_2rdm_elements(connected_det, connected_ind, this_det_rev, this_ind, 0.5);
        get_2rdm_elements(connected_det_rev, connected_ind, this_det, this_ind, 0.5);
        get_2rdm_elements(connected_det_rev, connected_ind, this_det_rev, this_ind, 0.5);
      }
    }
  }
}

void RDM::get_2rdm_elements(
    const Det& connected_det,
    const size_t j_det,
    const Det& this_det,
    const size_t i_det,
    const double tr_factor) {
  //=====================================================
  // Fill in 2RDM for a given pair of dets. When the two dets
  // are not idential also do the pair in reverse order since
  // the Hamiltonian is upper triangular.
  //
  // D_pqrs corresponds to <a^+_p a^+_q a_r a_s>
  // Four different spin configurations
  // (1) p: up, q: up, r: up, s: up
  // (2) p: dn, q: dn, r: dn, s: dn
  // (3) p: dn, q: up, r: up, s: dn
  // (4) p: up, q: dn, r: dn, s: up
  //
  // Created: Y. Yao, August 2018
  //=====================================================

  std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
  std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

  // 0 alpha excitation apart
  if (this_det.up == connected_det.up) {
    if (this_det.dn == connected_det.dn) {  // 0 beta excitation apart

      // (1)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        for (unsigned j_elec = i_elec + 1; j_elec < n_up; j_elec++) {
          unsigned s = occ_up[i_elec];
          unsigned r = occ_up[j_elec];

          write_in_2rdm(s, r, r, s, tr_factor, i_det, j_det);
          write_in_2rdm(s, r, s, r, -tr_factor, i_det, j_det);
          write_in_2rdm(r, s, s, r, tr_factor, i_det, j_det);
          write_in_2rdm(r, s, r, s, -tr_factor, i_det, j_det);
        }
      }

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        for (unsigned j_elec = i_elec + 1; j_elec < n_dn; j_elec++) {
          unsigned s = occ_dn[i_elec];
          unsigned r = occ_dn[j_elec];

          write_in_2rdm(s, r, r, s, tr_factor, i_det, j_det);
          write_in_2rdm(s, r, s, r, -tr_factor, i_det, j_det);
          write_in_2rdm(r, s, s, r, tr_factor, i_det, j_det);
          write_in_2rdm(r, s, r, s, -tr_factor, i_det, j_det);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        for (unsigned j_elec = 0; j_elec < n_dn; j_elec++) {
          unsigned s = occ_up[i_elec];
          unsigned r = occ_dn[j_elec];

          write_in_2rdm(s, r, r, s, tr_factor, i_det, j_det);
          write_in_2rdm(r, s, s, r, tr_factor, i_det, j_det);
        }
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // to

      // (2)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        if ((p != b1) && (p != b2)) {
          double signed_factor = tr_factor * permfac_ccaa(this_det.dn, p, b2, b1, p);

          write_in_2rdm(p, b2, b1, p, signed_factor, i_det, j_det);
          write_in_2rdm(b2, p, b1, p, -signed_factor,i_det, j_det);
          write_in_2rdm(p, b2, p, b1, -signed_factor,i_det, j_det);
          write_in_2rdm(b2, p, p, b1, signed_factor, i_det, j_det);

          signed_factor = tr_factor * permfac_ccaa(connected_det.dn, p, b1, b2, p);

          write_in_2rdm(p, b1, b2, p, signed_factor, i_det, j_det);
          write_in_2rdm(b1, p, b2, p, -signed_factor, i_det, j_det);
          write_in_2rdm(p, b1, p, b2, -signed_factor, i_det, j_det);
          write_in_2rdm(b1, p, p, b2, signed_factor, i_det, j_det);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up[i_elec];
        double signed_factor = tr_factor * connected_det.dn.diff(this_det.dn).permutation_factor;

        write_in_2rdm(p, b2, b1, p, signed_factor, i_det, j_det);
        write_in_2rdm(b2, p, p, b1, signed_factor, i_det, j_det);

        signed_factor = tr_factor * this_det.dn.diff(connected_det.dn).permutation_factor;

        write_in_2rdm(p, b1, b2, p, signed_factor, i_det, j_det);
        write_in_2rdm(b1, p, p, b2, signed_factor, i_det, j_det);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 2) {  // 2 beta excitations apart
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // from
      unsigned b2 = connected_det.dn.diff(this_det.dn).right_only[1];
      unsigned b3 = connected_det.dn.diff(this_det.dn).left_only[0];  // to
      unsigned b4 = connected_det.dn.diff(this_det.dn).left_only[1];

      double signed_factor = tr_factor * connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm(b3, b4, b2, b1, signed_factor, i_det, j_det);
      write_in_2rdm(b3, b4, b1, b2, -signed_factor,i_det, j_det);
      write_in_2rdm(b4, b3, b2, b1, -signed_factor,i_det, j_det);
      write_in_2rdm(b4, b3, b1, b2, signed_factor, i_det, j_det);

      signed_factor = tr_factor * this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm(b1, b2, b4, b3, signed_factor, i_det, j_det);
      write_in_2rdm(b1, b2, b3, b4, -signed_factor, i_det, j_det);
      write_in_2rdm(b2, b1, b4, b3, -signed_factor, i_det, j_det);
      write_in_2rdm(b2, b1, b3, b4, signed_factor, i_det, j_det);
    }

    // 1 alpha excitation apart
  } else if (connected_det.up.diff(this_det.up).n_diffs == 1) {
    if (this_det.dn == connected_det.dn) {  // 0 beta excitation apart
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // from
      unsigned a2 = connected_det.up.diff(this_det.up).left_only[0];  // to

      // (1)
      for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
        unsigned p = occ_up[i_elec];
        if ((p != a1) && (p != a2)) {
          double signed_factor = tr_factor * permfac_ccaa(this_det.up, p, a2, a1, p);

          write_in_2rdm(p, a2, a1, p, signed_factor, i_det, j_det);  
          write_in_2rdm(p, a2, p, a1, -signed_factor, i_det, j_det);
          write_in_2rdm(a2, p, a1, p, -signed_factor, i_det, j_det);
          write_in_2rdm(a2, p, p, a1, signed_factor, i_det, j_det);

          signed_factor = tr_factor * permfac_ccaa(connected_det.up, p, a1, a2, p);

          write_in_2rdm(p, a1, a2, p, signed_factor, i_det, j_det);
          write_in_2rdm(p, a1, p, a2, -signed_factor, i_det, j_det);
          write_in_2rdm(a1, p, a2, p, -signed_factor, i_det, j_det);
          write_in_2rdm(a1, p, p, a2, signed_factor, i_det, j_det);
        }
      }

      // (3) (4)
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];

        double signed_factor = tr_factor * connected_det.up.diff(this_det.up).permutation_factor;

        write_in_2rdm(a2, p, p, a1, signed_factor, i_det, j_det);
        write_in_2rdm(p, a2, a1, p, signed_factor, i_det, j_det);

        signed_factor = tr_factor * this_det.up.diff(connected_det.up).permutation_factor;

        write_in_2rdm(a1, p, p, a2, signed_factor, i_det, j_det);
        write_in_2rdm(p, a1, a2, p, signed_factor, i_det, j_det);
      }

    } else if (connected_det.dn.diff(this_det.dn).n_diffs == 1) {  // 1 beta excitation apart
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // up from
      unsigned a2 = connected_det.up.diff(this_det.up).left_only[0];  // up to
      unsigned b1 = connected_det.dn.diff(this_det.dn).right_only[0];  // dn from
      unsigned b2 = connected_det.dn.diff(this_det.dn).left_only[0];  // dn to

      double signed_factor = tr_factor * connected_det.up.diff(this_det.up).permutation_factor *
                       connected_det.dn.diff(this_det.dn).permutation_factor;

      write_in_2rdm(a2, b2, b1, a1, signed_factor, i_det, j_det);
      write_in_2rdm(b2, a2, a1, b1, signed_factor, i_det, j_det);

      signed_factor = tr_factor * this_det.up.diff(connected_det.up).permutation_factor *
                this_det.dn.diff(connected_det.dn).permutation_factor;

      write_in_2rdm(a1, b1, b2, a2, signed_factor, i_det, j_det);
      write_in_2rdm(b1, a1, a2, b2, signed_factor, i_det, j_det);
    }

    // 2 alpha excitations apart
  } else if (connected_det.up.diff(this_det.up).n_diffs == 2) {
    if (this_det.dn == connected_det.dn) {
      unsigned a1 = connected_det.up.diff(this_det.up).right_only[0];  // from
      unsigned a2 = connected_det.up.diff(this_det.up).right_only[1];
      unsigned a3 = connected_det.up.diff(this_det.up).left_only[0];  // to
      unsigned a4 = connected_det.up.diff(this_det.up).left_only[1];

      double signed_factor = tr_factor * connected_det.up.diff(this_det.up).permutation_factor;

      write_in_2rdm(a3, a4, a2, a1, signed_factor, i_det, j_det);
      write_in_2rdm(a3, a4, a1, a2, -signed_factor, i_det, j_det);
      write_in_2rdm(a4, a3, a2, a1, -signed_factor, i_det, j_det);
      write_in_2rdm(a4, a3, a1, a2, signed_factor, i_det, j_det);

      signed_factor = tr_factor * this_det.up.diff(connected_det.up).permutation_factor;

      write_in_2rdm(a1, a2, a4, a3, signed_factor, i_det, j_det);
      write_in_2rdm(a1, a2, a3, a4, -signed_factor, i_det, j_det);
      write_in_2rdm(a2, a1, a4, a3, -signed_factor, i_det, j_det);
      write_in_2rdm(a2, a1, a3, a4, signed_factor, i_det, j_det);
    }
  }
}

void RDM::write_in_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const double factor,
		const size_t i_det, const size_t j_det) {
  // <psi_{i_det}| c^+_p c^+_q c_r c_s | psi_{j_det}>
  // By symmetry (p,s)<->(q,r) only half of the 2RDM needs storing.
  size_t a = p * n_orbs + s;
  size_t b = q * n_orbs + r;
  if (a >= b) {
    double value = 0.;
    for (unsigned i_state=0; i_state<n_states; i_state++) value += coefs[i_state][i_det] * coefs[i_state][j_det];
    value *= factor;
#pragma omp atomic
    two_rdm[(a * (a + 1)) / 2 + b] += value;
  }    
}

void RDM::MPI_Allreduce_2rdm() {
  // MPI_Allreduce after computing on mutiple nodes
  if (Parallel::get_n_procs() > 1) {
    std::vector<double> global_two_rdm(two_rdm.size());

    double* src_ptr = two_rdm.data();
    double* dest_ptr = global_two_rdm.data();

    const size_t CHUNK_SIZE = 1 << 27;
    unsigned n_elems_left = two_rdm.size();

    while (n_elems_left > CHUNK_SIZE) {
      MPI_Allreduce(src_ptr, dest_ptr, CHUNK_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      n_elems_left -= CHUNK_SIZE;
      src_ptr += CHUNK_SIZE;
      dest_ptr += CHUNK_SIZE;
    }
    MPI_Allreduce(src_ptr, dest_ptr, n_elems_left, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    two_rdm = global_two_rdm;
  }
}

double RDM::one_rdm_elem(const unsigned p, const unsigned q) const { return one_rdm(p, q); }

double RDM::two_rdm_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  return two_rdm[combine4_2rdm(p, q, r, s)];
}

void RDM::get_1rdm_from_2rdm() {
  // construct 1rdm from 2rdm
  one_rdm = MatrixXd::Zero(n_orbs, n_orbs);
  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned s = 0; s < n_orbs; s++) {
      for (unsigned k = 0; k < n_orbs; k++) {
        one_rdm(p, s) += two_rdm_elem(p, k, k, s) / (1. * (n_up + n_dn) - 1.);
      }
    }
  }

  Timer::checkpoint("get 1rdm from 2rdm");
}

void RDM::clear() {
  one_rdm.resize(0, 0);
  two_rdm.clear();
  two_rdm.shrink_to_fit();
  
  // Clear spin-resolved RDMs
  spin_one_rdm_up.resize(0, 0);
  spin_one_rdm_dn.resize(0, 0);
  spin_two_rdm_aa.clear();
  spin_two_rdm_bb.clear();
  spin_two_rdm_ab.clear();
  spin_two_rdm_ba.clear();
  spin_two_rdm_aa.shrink_to_fit();
  spin_two_rdm_bb.shrink_to_fit();
  spin_two_rdm_ab.shrink_to_fit();
  spin_two_rdm_ba.shrink_to_fit();
}

void RDM::compute_energy_from_rdm() const {
  //=====================================================
  // Reproduce variational energy from 2RDM.
  // Currently not used in the program;
  // Useful tool for verifying the 2RDM.
  //
  // Created: Y. Yao, July 2018
  //=====================================================

  MatrixXd tmp_1rdm = MatrixXd::Zero(n_orbs, n_orbs);

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned s = 0; s < n_orbs; s++) {
      for (unsigned k = 0; k < n_orbs; k++) {
        tmp_1rdm(p, s) += two_rdm_elem(p, k, k, s) / (1. * (n_up + n_dn) - 1.);
      }
    }
  }

  std::cout << "tmp_1rdm\n" << tmp_1rdm << "\n";
  std::cout << "one_rdm\n" << one_rdm << "\n";

  double onebody = 0., twobody = 0.;

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      onebody += tmp_1rdm(p, q) * integrals.get_1b(p, q);
    }
  }

  for (unsigned p = 0; p < n_orbs; p++) {
    for (unsigned q = 0; q < n_orbs; q++) {
      for (unsigned r = 0; r < n_orbs; r++) {
        for (unsigned s = 0; s < n_orbs; s++) {
          twobody += 0.5 * two_rdm_elem(p, q, r, s) * integrals.get_2b(p, s, q, r);
        }
      }
    }
  }

  std::printf(
      "Energy:\ncore: %.10f\none-body: %.10f\ntwo-body: %.10f\ntotal: %.10f\n",
      integrals.energy_core,
      onebody,
      twobody,
      onebody + twobody + integrals.energy_core);
}

//=====================================================
// Spin-resolved RDM implementations
//=====================================================

void RDM::get_spin_1rdm() {
  //=====================================================
  // Create spin-resolved 1RDM using the variational wavefunction
  //
  // γ^α(p,q) = ⟨Ψ|a†_{p,α} a_{q,α}|Ψ⟩
  // γ^β(p,q) = ⟨Ψ|a†_{p,β} a_{q,β}|Ψ⟩
  //
  // Created: Based on existing get_1rdm() implementation
  //=====================================================
  Timer::start("get spin 1rdm");
  const std::vector<unsigned>& orb_sym = integrals.orb_sym;

  spin_one_rdm_up = MatrixXd::Zero(n_orbs, n_orbs);
  spin_one_rdm_dn = MatrixXd::Zero(n_orbs, n_orbs);

  // Create hash table; used for looking up the index of a det
  std::unordered_map<Det, size_t, DetHasher> det2ind;
  for (size_t i = 0; i < dets.size(); i++) {
    det2ind[dets[i]] = i;
  }

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < dets.size(); i_det++) {
    const Det& this_det = dets[i_det];

    const std::vector<unsigned> occ_up = this_det.up.get_occupied_orbs();
    const std::vector<unsigned> occ_dn = this_det.dn.get_occupied_orbs();

    // Spin-up electrons
    for (unsigned i_elec = 0; i_elec < n_up; i_elec++) {
      unsigned p = occ_up[i_elec];
      for (unsigned r = 0; r < n_orbs; r++) {
        if (orb_sym[p] != orb_sym[r]) continue;
        if (p != r && this_det.up.has(r)) continue;

        Det new_det = this_det;
        new_det.up.unset(p);
        new_det.up.set(r);

        size_t connected_ind;
        double factor = 1.;
        if (!time_sym) {
          if (det2ind.count(new_det) == 1) {
            connected_ind = det2ind[new_det];
          } else
            continue;
        } else {
          Det new_det_proxy = new_det;
          if (new_det.up > new_det.dn)
            new_det_proxy.reverse_spin();
          if (det2ind.count(new_det_proxy) == 1) {
            connected_ind = det2ind[new_det_proxy];
          } else
            continue;
          if (this_det.up != this_det.dn) factor *= Util::SQRT2_INV;
          if (new_det.up != new_det.dn) factor *= Util::SQRT2_INV;
        }
        factor *= this_det.up.diff(new_det.up).permutation_factor;

        write_in_spin_1rdm(p, r, factor, i_det, connected_ind, true); // true = spin-up
      }
    }

    // Spin-down electrons
    if (!time_sym) {
      for (unsigned i_elec = 0; i_elec < n_dn; i_elec++) {
        unsigned p = occ_dn[i_elec];
        for (unsigned r = 0; r < n_orbs; r++) {
          if (orb_sym[p] != orb_sym[r]) continue;
          if (p != r && this_det.dn.has(r)) continue;

          Det new_det = this_det;
          new_det.dn.unset(p);
          new_det.dn.set(r);

          if (det2ind.count(new_det) == 0)
            continue;

          write_in_spin_1rdm(p, r, this_det.dn.diff(new_det.dn).permutation_factor, 
              i_det, det2ind[new_det], false); // false = spin-down
        }
      }
    }
  }

  if (time_sym) {
    spin_one_rdm_up *= 2.;
    spin_one_rdm_dn = spin_one_rdm_up; // For time_sym, up and down are identical
  }
  
  Timer::end();
}

void RDM::write_in_spin_1rdm(const unsigned p, const unsigned q, const double factor, 
        const size_t i_det, const size_t j_det, const bool is_up) {
  // ⟨ψ_{i_det}| c^†_{p,σ} c_{q,σ} |ψ_{j_det}⟩ where σ = α (up) or β (down)
  double value = 0.;
  for (unsigned i_state = 0; i_state < n_states; i_state++) 
    value += coefs[i_state][i_det] * coefs[i_state][j_det];
  value *= factor;
  
  if (is_up) {
#pragma omp atomic
    spin_one_rdm_up(p, q) += value;
  } else {
#pragma omp atomic
    spin_one_rdm_dn(p, q) += value;
  }
}

double RDM::spin_one_rdm_elem(const unsigned p, const unsigned q, const bool is_up) const {
  if (is_up) {
    return spin_one_rdm_up(p, q);
  } else {
    return spin_one_rdm_dn(p, q);
  }
}

void RDM::dump_spin_1rdm() const {
  if (Parallel::is_master()) {
    FILE* pFile;
    
    // Dump spin-up 1RDM
    pFile = fopen("spin_1rdm_up.csv", "w");
    fprintf(pFile, "p,r,spin_1rdm_up\n");
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned r = p; r < n_orbs; r++) {
        const double rdm_pr = spin_one_rdm_up(p, r);
        if (std::abs(rdm_pr) < 1e-9) continue;
        fprintf(pFile, "%d,%d,%#.15g\n", integrals.orb_order[p], integrals.orb_order[r], rdm_pr);
      }
    }
    fclose(pFile);
    
    // Dump spin-down 1RDM
    pFile = fopen("spin_1rdm_dn.csv", "w");
    fprintf(pFile, "p,r,spin_1rdm_dn\n");
    for (unsigned p = 0; p < n_orbs; p++) {
      for (unsigned r = p; r < n_orbs; r++) {
        const double rdm_pr = spin_one_rdm_dn(p, r);
        if (std::abs(rdm_pr) < 1e-9) continue;
        fprintf(pFile, "%d,%d,%#.15g\n", integrals.orb_order[p], integrals.orb_order[r], rdm_pr);
      }
    }
    fclose(pFile);
  }
}


void RDM::get_spin_2rdm(const std::vector<std::vector<size_t>>& connections) {
  Timer::start("get spin 2rdm");
  const size_t n_orbs_ll = n_orbs;
  const size_t size_two_rdm = n_orbs_ll * n_orbs_ll * (n_orbs_ll * n_orbs_ll + 1) / 2;
  
  spin_two_rdm_aa.resize(size_two_rdm, 0.);
  spin_two_rdm_bb.resize(size_two_rdm, 0.);
  spin_two_rdm_ab.resize(size_two_rdm, 0.);
  spin_two_rdm_ba.resize(size_two_rdm, 0.);

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < connections.size(); i_det++) {
    Det this_det = dets[i_det];
    for (size_t j_det = 0; j_det < connections[i_det].size(); j_det++) {
      size_t connected_ind = connections[i_det][j_det];
      Det connected_det = dets[connected_ind];
      get_spin_2rdm_pair(connected_det, connected_ind, this_det, i_det);
    }
  }
  MPI_Allreduce_spin_2rdm();
  Timer::end();
}

void RDM::get_spin_2rdm(const SparseMatrix& hamiltonian_matrix) {
  Timer::start("get spin 2rdm");
  const size_t n_orbs_ll = n_orbs;
  const size_t size_two_rdm = n_orbs_ll * n_orbs_ll * (n_orbs_ll * n_orbs_ll + 1) / 2;
  
  spin_two_rdm_aa.resize(size_two_rdm, 0.);
  spin_two_rdm_bb.resize(size_two_rdm, 0.);
  spin_two_rdm_ab.resize(size_two_rdm, 0.);
  spin_two_rdm_ba.resize(size_two_rdm, 0.);

#pragma omp parallel for schedule(dynamic, 10)
  for (size_t i_det = 0; i_det < hamiltonian_matrix.count_n_rows(); i_det++) {
    Det this_det = dets[i_det];
    auto hamiltonian_row = hamiltonian_matrix.get_row(i_det);
    for (size_t j_det = 0; j_det < hamiltonian_row.size(); j_det++) {
      size_t connected_ind = hamiltonian_row.get_index(j_det);
      Det connected_det = dets[connected_ind];
      get_spin_2rdm_pair(connected_det, connected_ind, this_det, i_det);
    }
  }
  MPI_Allreduce_spin_2rdm();
  Timer::end();
}

inline void RDM::get_spin_2rdm_pair(const Det& connected_det, 
       const size_t connected_ind, const Det& this_det, const size_t this_ind) {
  if (!time_sym)
    get_spin_2rdm_elements(connected_det, connected_ind, this_det, this_ind, 1.);
  else {
    if (this_det.up == this_det.dn) {
      if (connected_det.up == connected_det.dn) {
        get_spin_2rdm_elements(connected_det, connected_ind, this_det, this_ind, 1.);
      } else {
        Det connected_det_rev = connected_det;
        connected_det_rev.reverse_spin();
        get_spin_2rdm_elements(connected_det, connected_ind, this_det, this_ind, Util::SQRT2_INV);
        get_spin_2rdm_elements(connected_det_rev, connected_ind, this_det, this_ind, Util::SQRT2_INV);
      }
    } else {
      get_spin_2rdm_elements(connected_det, connected_ind, this_det, this_ind, Util::SQRT2_INV);
    }
  }
}

void RDM::get_spin_2rdm_elements(const Det& connected_det, const size_t j_det, 
        const Det& this_det, const size_t i_det, const double tr_factor) {
  const auto& diff_up = connected_det.up.diff(this_det.up);
  const auto& diff_dn = connected_det.dn.diff(this_det.dn);
  
  if (diff_up.n_diffs == 2 && diff_dn.n_diffs == 0) {
    unsigned a1 = diff_up.left_only[0], a2 = diff_up.left_only[1];
    unsigned a3 = diff_up.right_only[0], a4 = diff_up.right_only[1];
    double signed_factor = tr_factor * connected_det.up.diff(this_det.up).permutation_factor;
    write_in_spin_2rdm(a3, a4, a2, a1, signed_factor, i_det, j_det, "aa");
    write_in_spin_2rdm(a3, a4, a1, a2, -signed_factor, i_det, j_det, "aa");
    write_in_spin_2rdm(a4, a3, a2, a1, -signed_factor, i_det, j_det, "aa");
    write_in_spin_2rdm(a4, a3, a1, a2, signed_factor, i_det, j_det, "aa");
  } else if (diff_up.n_diffs == 0 && diff_dn.n_diffs == 2) {
    unsigned b1 = diff_dn.left_only[0], b2 = diff_dn.left_only[1];
    unsigned b3 = diff_dn.right_only[0], b4 = diff_dn.right_only[1];
    double signed_factor = tr_factor * connected_det.dn.diff(this_det.dn).permutation_factor;
    write_in_spin_2rdm(b3, b4, b2, b1, signed_factor, i_det, j_det, "bb");
    write_in_spin_2rdm(b3, b4, b1, b2, -signed_factor, i_det, j_det, "bb");
    write_in_spin_2rdm(b4, b3, b2, b1, -signed_factor, i_det, j_det, "bb");
    write_in_spin_2rdm(b4, b3, b1, b2, signed_factor, i_det, j_det, "bb");
  } else if (diff_up.n_diffs == 1 && diff_dn.n_diffs == 1) {
    unsigned a1 = diff_up.left_only[0], a2 = diff_up.right_only[0];
    unsigned b1 = diff_dn.left_only[0], b2 = diff_dn.right_only[0];
    double signed_factor = tr_factor * connected_det.up.diff(this_det.up).permutation_factor * 
                          connected_det.dn.diff(this_det.dn).permutation_factor;
    write_in_spin_2rdm(a2, b2, b1, a1, signed_factor, i_det, j_det, "ab");
    write_in_spin_2rdm(b2, a2, a1, b1, signed_factor, i_det, j_det, "ba");
  }
}

void RDM::write_in_spin_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s, 
        const double factor, const size_t i_det, const size_t j_det, const std::string& spin_config) {
  double value = 0.;
  for (unsigned i_state = 0; i_state < n_states; i_state++) 
    value += coefs[i_state][i_det] * coefs[i_state][j_det];
  value *= factor;
  
  size_t index = combine4_spin_2rdm(p, q, r, s);
  
  if (spin_config == "aa") {
#pragma omp atomic
    spin_two_rdm_aa[index] += value;
  } else if (spin_config == "bb") {
#pragma omp atomic
    spin_two_rdm_bb[index] += value;
  } else if (spin_config == "ab") {
#pragma omp atomic
    spin_two_rdm_ab[index] += value;
  } else if (spin_config == "ba") {
#pragma omp atomic
    spin_two_rdm_ba[index] += value;
  }
}

inline size_t RDM::combine4_spin_2rdm(const unsigned p, const unsigned q, const unsigned r, const unsigned s) const {
  const size_t a = p * n_orbs + s;
  const size_t b = q * n_orbs + r;
  return (a >= b) ? (a * (a + 1)) / 2 + b : (b * (b + 1)) / 2 + a;
}

double RDM::spin_two_rdm_elem(const unsigned p, const unsigned q, const unsigned r, const unsigned s, const std::string& spin_config) const {
  size_t index = combine4_spin_2rdm(p, q, r, s);
  if (spin_config == "aa") return spin_two_rdm_aa[index];
  else if (spin_config == "bb") return spin_two_rdm_bb[index];
  else if (spin_config == "ab") return spin_two_rdm_ab[index];
  else if (spin_config == "ba") return spin_two_rdm_ba[index];
  return 0.0;
}

void RDM::MPI_Allreduce_spin_2rdm() {
  if (Parallel::get_n_procs() > 1) {
    const size_t CHUNK_SIZE = 1 << 27;
    auto reduce_vector = [&](std::vector<double>& vec) {
      std::vector<double> global_vec(vec.size());
      double* src = vec.data();
      double* dest = global_vec.data();
      unsigned n_left = vec.size();
      while (n_left > CHUNK_SIZE) {
        MPI_Allreduce(src, dest, CHUNK_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        n_left -= CHUNK_SIZE;
        src += CHUNK_SIZE;
        dest += CHUNK_SIZE;
      }
      MPI_Allreduce(src, dest, n_left, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      vec = global_vec;
    };
    reduce_vector(spin_two_rdm_aa);
    reduce_vector(spin_two_rdm_bb);
    reduce_vector(spin_two_rdm_ab);
    reduce_vector(spin_two_rdm_ba);
  }
}

void RDM::dump_spin_2rdm(const bool dump_csv) const {
  if (Parallel::is_master()) {
    std::cout << "writing out spin-resolved 2RDM\n";
    if (dump_csv) {
      auto dump_spin_config = [&](const std::string& filename, const std::string& config) {
        FILE* pFile = fopen(filename.c_str(), "w");
        fprintf(pFile, "p,q,r,s,spin_2rdm_%s\n", config.c_str());
        for (unsigned p = 0; p < n_orbs; p++) {
          for (unsigned q = 0; q < n_orbs; q++) {
            for (unsigned r = 0; r < n_orbs; r++) {
              for (unsigned s = 0; s < n_orbs; s++) {
                const double rdm_pqrs = spin_two_rdm_elem(p, q, r, s, config);
                if (std::abs(rdm_pqrs) < 1.0e-9) continue;
                fprintf(pFile, "%d,%d,%d,%d,%#.15g\n", 
                        integrals.orb_order[p], integrals.orb_order[q], 
                        integrals.orb_order[r], integrals.orb_order[s], rdm_pqrs);
              }
            }
          }
        }
        fclose(pFile);
      };
      dump_spin_config("spin_2rdm_aa.csv", "aa");
      dump_spin_config("spin_2rdm_bb.csv", "bb");
      dump_spin_config("spin_2rdm_ab.csv", "ab");
      dump_spin_config("spin_2rdm_ba.csv", "ba");
    }
  }
}
