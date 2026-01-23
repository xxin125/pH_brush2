#ifndef MC_HPP
#define MC_HPP

#include <vector>
#include <random>
#include <string>
#include <thrust/device_vector.h>
#include "types.hpp"
#include "device_data.hpp"
#include "pme.hpp"
#include "neighbor.hpp"

struct MCPlan {
    std::vector<int> idx_nh_n;
    std::vector<int> idx_nh_p;
    std::vector<int> idx_w;
    // z_cut removed (not used in exact GPU proposal)
};

struct PhaseGCResult {
    int attempted{0};
    int accepted{0};
    int accepted_forward{0};
    int accepted_backward{0};
    int Nh{0};
    int Nprot{0};
    real E{0.0};
    real mu_eff{0.0};
};

MCPlan build_plan(const System& sys, const Params& p);

// Build device resources for GPU-based proposal (call once per neighbor rebuild)
DeviceCSR build_device_move_neigh_from_host(const HostNeigh& h_neigh);
thrust::device_vector<int> build_device_nh_sites(const MCPlan& mcplan);

// Constant-pH SGMC with exact detailed balance (GPU proposal)
// d_neigh_energy: half neighbor list for energy calculation
// d_neigh_move: full symmetric neighbor list for proposal
PhaseGCResult phase_gc_const_pH_exact_gpu(System& sys,
                                          DeviceAtoms& d_atoms,
                                          const DeviceParams& d_params,
                                          const DeviceCSR& d_neigh_energy,
                                          const DeviceCSR& d_neigh_move,
                                          const Box& box,
                                          PMEPlan* plan,
                                          const Params& p,
                                          const MCPlan& mcplan,
                                          thrust::device_vector<int>& d_nh_sites,
                                          std::mt19937& rng);

#endif // MC_HPP
