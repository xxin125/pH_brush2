#pragma once
#include <vector>
#include <string>
#include "precision.hpp"

struct Box {
    real Lx{0}, Ly{0}, Lz{0};
};

struct Atom {
    int id{0};
    int mol{0};
    int type{0};
    real q{0.0};
    real x{0}, y{0}, z{0};
};

struct Bond {
    int id{0};
    int type{0};
    int i{0}; // 0-based atom index
    int j{0}; // 0-based atom index
};

struct System {
    Box box;
    std::vector<Atom> atoms;
    std::vector<Bond> bonds;
    int natoms{0};
    int nbonds{0};
    int ntypes{0};
};

struct Params {
    real rc{0.0};
    int ntypes{0};
    std::vector<real> lj_eps; // size ntypes*ntypes
    std::vector<real> lj_sig; // size ntypes*ntypes
    bool lj_shift{true};        // potential-shift to zero at rc
    std::vector<real> lj_shift_tbl; // precomputed LJ(rc) per pair, size ntypes*ntypes
    // Coulomb
    std::string coulombtype{"coul_cut"}; // "coul_cut" or "pme"
    real epsilon_r{1.0};      // relative dielectric constant
    real ewald_rtol{real(1e-5)};
    real ewald_alpha{0.0};
    real pme_spacing{0.12};
    int pme_order{4};
    bool pme_3dc_enabled{false};
    real pme_3dc_zfac{3.0};

    int NH_N_type{4};
    int NH_P_type{5};
    int W_type{7};
    int Cl_type{8};
    int rng_seed{2025};
    int energy_interval{100};
    real beta{real(1.0 / (0.008314462 * 300.0))};
    real w_z_extra_nm{1.0};
    real calibration_offset{0.0}; // correction applied to protonation moves
    

    // Constant-pH SGMC
    real pH{0.0};          // external pH
    real pKa_NH{0.0};      // intrinsic pKa for NH sites
    int  cph_attempts{0};  // total attempts for const-pH SGMC
    real rc_move{0.0};     // cutoff for MC proposal neighbors (0 = use rc)
};

struct CSR {
    std::vector<int> head; // size N+1
    std::vector<int> list; // size head[N]
};
