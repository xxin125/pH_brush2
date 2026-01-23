#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include "mc.hpp"
#include "energy.hpp"
#include "report.hpp"

// ============================================================================
// CUDA error checking macros
// ============================================================================

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUDA_KERNEL_CHECK() do { \
    CUDA_CHECK(cudaGetLastError()); \
} while(0)

// ============================================================================
// Debug: check host/device consistency (only in debug builds)
// ============================================================================

#ifndef NDEBUG
static void check_host_device_consistency(const System& sys,
                                          const DeviceAtoms& d_atoms,
                                          int sample_count = 32,
                                          std::mt19937* rng = nullptr)
{
    const int N = sys.natoms;
    if (N == 0) return;
    
    std::vector<int> indices;
    indices.reserve(std::min(N, sample_count));
    if (rng && sample_count < N) {
        // Shuffle-based unique sampling (no duplicates)
        indices.resize(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), *rng);
        indices.resize(sample_count);
    } else {
        // Check all (or first sample_count)
        for (int i = 0; i < std::min(N, sample_count); ++i) {
            indices.push_back(i);
        }
    }
    
    for (int i : indices) {
        int h_type = sys.atoms[i].type - 1;  // host is 1-based, device is 0-based
        real h_q = sys.atoms[i].q;
        
        int d_type = d_atoms.type[i];  // implicit D2H (intentional for debug)
        real d_q = d_atoms.q[i];
        
        if (h_type != d_type) {
            throw std::runtime_error("Host/device type mismatch at atom " + std::to_string(i) 
                                     + ": host=" + std::to_string(h_type) 
                                     + ", device=" + std::to_string(d_type));
        }
        if (std::abs(h_q - d_q) > (real)1e-6) {
            throw std::runtime_error("Host/device charge mismatch at atom " + std::to_string(i));
        }
    }
}
#endif

// ============================================================================
// Device helper: apply/restore atom identity on GPU without cudaMemcpy
// bad_flag: device pointer to int, set to 1 on out-of-bounds (for host to check)
// ============================================================================

__global__ void k_apply_move(int i, int j, int s, int N,
                             int* __restrict__ type,
                             real* __restrict__ q,
                             int t_nh_n, int t_nh_p, int t_w, int t_cl,
                             int* __restrict__ bad_flag)
{
    // Boundary check: signal error via bad_flag instead of silent return
    if (i < 0 || i >= N || j < 0 || j >= N){
        if (bad_flag) atomicExch(bad_flag, 1);
        return;
    }
    
    if (s == +1){
        type[i] = t_nh_p; q[i] = (real)+1.0;
        type[j] = t_cl;   q[j] = (real)-1.0;
    } else {
        type[i] = t_nh_n; q[i] = (real)0.0;
        type[j] = t_w;    q[j] = (real)0.0;
    }
}

__global__ void k_restore_two(int i, int ti, real qi,
                              int j, int tj, real qj, int N,
                              int* __restrict__ type,
                              real* __restrict__ q,
                              int* __restrict__ bad_flag)
{
    // Boundary check: signal error via bad_flag instead of silent return
    if (i < 0 || i >= N || j < 0 || j >= N){
        if (bad_flag) atomicExch(bad_flag, 1);
        return;
    }
    
    type[i] = ti; q[i] = qi;
    type[j] = tj; q[j] = qj;
}

// ============================================================================
// GPU kernels for move counting and selection
// ============================================================================

// Count valid moves for each NH site
__global__ void k_count_moves_per_nh(const int* __restrict__ nh_sites, int M, int N,
                                     const int* __restrict__ head,
                                     const int* __restrict__ list,
                                     const int* __restrict__ type,
                                     int t_nh_n, int t_nh_p, int t_w, int t_cl,
                                     long long* __restrict__ counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    int i  = nh_sites[tid];
    
    // Boundary check: if i is invalid, output 0 count (host will get K=0 or wrong result)
    if (i < 0 || i >= N) { counts[tid] = 0; return; }
    
    int ti = type[i];

    int target = -1;
    if      (ti == t_nh_n) target = t_w;   // forward: need W neighbor
    else if (ti == t_nh_p) target = t_cl;  // backward: need Cl neighbor
    else { counts[tid] = 0; return; }

    long long c = 0;
    for (int kk = head[i]; kk < head[i+1]; ++kk){
        int j = list[kk];
        if (type[j] == target) ++c;
    }
    counts[tid] = c;
}

// Find j and s for a chosen NH i and local index (0..count-1)
__global__ void k_pick_j_for_i(int i, long long local, int N,
                               const int* __restrict__ head,
                               const int* __restrict__ list,
                               const int* __restrict__ type,
                               int t_nh_n, int t_nh_p, int t_w, int t_cl,
                               int* __restrict__ out_j,
                               int* __restrict__ out_s)
{
    // Boundary check: if i is invalid, return error values (host will throw)
    if (i < 0 || i >= N) { *out_j = -1; *out_s = 0; return; }
    
    int ti = type[i];
    int target = -1;
    int s = 0;
    if      (ti == t_nh_n){ target = t_w;  s = +1; }
    else if (ti == t_nh_p){ target = t_cl; s = -1; }
    else { *out_j = -1; *out_s = 0; return; }

    long long c = 0;
    for (int kk = head[i]; kk < head[i+1]; ++kk){
        int j = list[kk];
        if (type[j] == target){
            if (c == local){
                *out_j = j;
                *out_s = s;
                return;
            }
            ++c;
        }
    }
    *out_j = -1;
    *out_s = 0;
}

// ============================================================================
// Host helpers: CSR conversion and device resource building
// ============================================================================

// Convert HostNeigh (vector<vector<int>>) -> host CSR
// Sort + unique to avoid duplicate edges
// Takes const& to avoid copying entire neighbor list
static CSR hostneigh_to_csr_symmetric(const HostNeigh& hn){
    const int N = (int)hn.nb.size();
    CSR out;
    out.head.resize(N + 1, 0);

    // First pass: count sizes after sort+unique
    long long total = 0;
    for (int i = 0; i < N; ++i){
        std::vector<int> v = hn.nb[i];  // copy per-row
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        out.head[i] = (int)total;
        total += (long long)v.size();
    }
    out.head[N] = (int)total;
    out.list.resize((size_t)total);

    // Second pass: fill list
    for (int i = 0; i < N; ++i){
        std::vector<int> v = hn.nb[i];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        int off = out.head[i];
        for (int k = 0; k < (int)v.size(); ++k){
            out.list[off + k] = v[k];
        }
    }
    return out;
}

// Build device CSR from host symmetric neighbor (for proposal, not energy)
DeviceCSR build_device_move_neigh_from_host(const HostNeigh& h_neigh){
    CSR csr = hostneigh_to_csr_symmetric(h_neigh);
    return copy_csr_to_device(csr);
}

// Build device array of all NH site indices (fixed set)
thrust::device_vector<int> build_device_nh_sites(const MCPlan& mcplan){
    std::vector<int> h;
    h.reserve(mcplan.idx_nh_n.size() + mcplan.idx_nh_p.size());
    h.insert(h.end(), mcplan.idx_nh_n.begin(), mcplan.idx_nh_n.end());
    h.insert(h.end(), mcplan.idx_nh_p.begin(), mcplan.idx_nh_p.end());

    thrust::device_vector<int> d = h;
    return d;
}

// ============================================================================
// GPU-based |M(x)| computation and move selection
// ============================================================================

// Compute K = |M(x)| and prefix-sum offsets for rank-based sampling
static long long compute_K_and_offsets(const thrust::device_vector<int>& d_nh_sites,
                                       const DeviceCSR& d_neigh_move,
                                       const thrust::device_vector<int>& d_type,
                                       int N,  // total atoms for boundary check
                                       int t_nh_n, int t_nh_p, int t_w, int t_cl,
                                       thrust::device_vector<long long>& d_counts,
                                       thrust::device_vector<long long>& d_off)
{
    const int M = (int)d_nh_sites.size();
    if (M == 0) return 0;
    
    d_counts.resize(M);
    d_off.resize(M + 1);

    int threads = 256;
    int blocks  = (M + threads - 1) / threads;

    k_count_moves_per_nh<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_nh_sites.data()), M, N,
        thrust::raw_pointer_cast(d_neigh_move.head.data()),
        thrust::raw_pointer_cast(d_neigh_move.list.data()),
        thrust::raw_pointer_cast(d_type.data()),
        t_nh_n, t_nh_p, t_w, t_cl,
        thrust::raw_pointer_cast(d_counts.data())
    );
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // exclusive scan counts -> off[0..M-1]
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_off.begin());

    // K = off[M-1] + counts[M-1]
    long long last_off = d_off[M - 1];
    long long last_cnt = d_counts[M - 1];
    long long K = last_off + last_cnt;

#ifndef NDEBUG
    // Debug sanity check
    if (K < 0) {
        throw std::runtime_error("compute_K_and_offsets: K < 0, data corruption suspected");
    }
#endif

    // store K to d_off[M] for upper_bound convenience
    d_off[M] = K;
    return K;
}

// Pick (i, j, s) for rank r using prefix-sum offsets
// d_j/d_s are pre-allocated buffers (reused, not allocated each step)
static void pick_move_rank_gpu(long long r,
                               const thrust::device_vector<int>& d_nh_sites,
                               const thrust::device_vector<long long>& d_off,
                               const DeviceCSR& d_neigh_move,
                               const thrust::device_vector<int>& d_type,
                               int N,  // total atoms for boundary check
                               int t_nh_n, int t_nh_p, int t_w, int t_cl,
                               thrust::device_vector<int>& d_j,
                               thrust::device_vector<int>& d_s,
                               int& out_i, int& out_j, int& out_s)
{
    const int M = (int)d_nh_sites.size();
    if (M == 0){
        throw std::runtime_error("pick_move_rank_gpu: M==0, no NH sites");
    }

    // Search in [0, M+1) so off[M]=K is included for robustness
    auto it = thrust::upper_bound(d_off.begin(), d_off.end(), r);
    int k = int(it - d_off.begin()) - 1;
    if (k < 0 || k >= M){
        throw std::runtime_error("pick_move_rank_gpu: upper_bound produced invalid k=" 
                                 + std::to_string(k) + " for r=" + std::to_string(r));
    }

    long long offk  = d_off[k];
    long long local = r - offk;
    if (local < 0){
        throw std::runtime_error("pick_move_rank_gpu: local<0, logic error");
    }

    // Get i from device (single element implicit D2H, acceptable)
    int i = d_nh_sites[k];

    // Pick j,s on device (reuse pre-allocated d_j/d_s)
    k_pick_j_for_i<<<1,1>>>(
        i, local, N,
        thrust::raw_pointer_cast(d_neigh_move.head.data()),
        thrust::raw_pointer_cast(d_neigh_move.list.data()),
        thrust::raw_pointer_cast(d_type.data()),
        t_nh_n, t_nh_p, t_w, t_cl,
        thrust::raw_pointer_cast(d_j.data()),
        thrust::raw_pointer_cast(d_s.data())
    );
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    out_i = i;
    out_j = d_j[0];
    out_s = d_s[0];

    // Must not fail - if it does, there's a logic bug that must be fixed
    if (out_j < 0 || (out_s != +1 && out_s != -1)){
        throw std::runtime_error("pick_move_rank_gpu: invalid (j=" + std::to_string(out_j) 
                                 + ", s=" + std::to_string(out_s) + ") after kernel. Logic bug.");
    }
}

// ============================================================================
// Energy calculation
// ============================================================================

struct AtomState { int type; real q; };

static inline real total_energy(const DeviceAtoms& d_atoms,
                                const DeviceParams& d_params,
                                const DeviceCSR& d_neigh,
                                const Box& box,
                                PMEPlan* plan,
                                PMEEnergyComponents* comps=nullptr){
    real E_lj = compute_lj_sr_energy_gpu(d_atoms, d_params, d_neigh, box);
    real E_c = 0.0;
    if (d_params.coulombtype == "coul_cut"){
        E_c = compute_coul_sr_energy_gpu(d_atoms, d_params, d_neigh, box);
    } else if (d_params.coulombtype == "pme"){
        if (!plan) throw std::runtime_error("total_energy: PME plan is null");
        E_c = pmeEnergy(*plan, d_atoms, d_neigh, box, comps);
    } else {
        throw std::runtime_error("total_energy: unsupported coulombtype");
    }
    return E_lj + E_c;
}

// ============================================================================
// MC plan building
// ============================================================================

MCPlan build_plan(const System& sys, const Params& p){
    MCPlan out;
    const int N = sys.natoms;
    out.idx_nh_n.reserve(N);
    out.idx_nh_p.reserve(N);
    out.idx_w.reserve(N);

    for (int i = 0; i < N; ++i){
        int t = sys.atoms[i].type;
        if      (t == p.NH_N_type) out.idx_nh_n.push_back(i);
        else if (t == p.NH_P_type) out.idx_nh_p.push_back(i);
        else if (t == p.W_type)    out.idx_w.push_back(i);
    }

    int nh_total = static_cast<int>(out.idx_nh_n.size() + out.idx_nh_p.size());
    if (nh_total == 0) throw std::runtime_error("[plan] No NH sites found.");

    mc_report() << "\n--- Plan summary (const-pH, EXACT-GPU) ---\n";
    mc_report() << "[NH] total=" << nh_total
                << ", NH_N=" << out.idx_nh_n.size()
                << ", NH_P=" << out.idx_nh_p.size() << "\n";
    mc_report() << "[W] W_total=" << out.idx_w.size() << "\n\n" << std::endl;
    return out;
}

// ============================================================================
// Main CpH SGMC function: Exact MH with GPU proposal (no pools, no z_cut)
// ============================================================================

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
                                          std::mt19937& rng)
{
    PhaseGCResult R{};

    const int M = (int)d_nh_sites.size();
    R.Nh = M;
    if (M == 0){
        mc_report() << "[cph] No titratable sites; nothing to do.\n";
        return R;
    }

#ifndef NDEBUG
    // Debug: verify host/device consistency at start
    check_host_device_consistency(sys, d_atoms, 32, &rng);
#endif

    // Current Nprot from host mirror
    {
        std::vector<int> h_nh;
        h_nh.reserve(mcplan.idx_nh_n.size() + mcplan.idx_nh_p.size());
        h_nh.insert(h_nh.end(), mcplan.idx_nh_n.begin(), mcplan.idx_nh_n.end());
        h_nh.insert(h_nh.end(), mcplan.idx_nh_p.begin(), mcplan.idx_nh_p.end());
        for (int i : h_nh){
            if (sys.atoms[i].type == p.NH_P_type) ++R.Nprot;
        }
    }

    R.E = total_energy(d_atoms, d_params, d_neigh_energy, box, plan);

    const real LN10 = std::log((real)10.0);
    const real mu_over_kT = LN10 * (p.pKa_NH - p.pH);
    R.mu_eff = mu_over_kT / p.beta;

    mc_report() << "[cph] init(EXACT-GPU): Nh=" << R.Nh
                << " Nprot=" << R.Nprot
                << " pH=" << p.pH
                << " pKa=" << p.pKa_NH
                << " mu/kT=" << mu_over_kT
                << " offset=" << p.calibration_offset
                << " E=" << R.E << std::endl;

    const int Amax = (p.cph_attempts > 0 ? p.cph_attempts : 100000);

    // Device-side temp buffers (pre-allocated, reused each step)
    thrust::device_vector<long long> d_counts, d_off;
    thrust::device_vector<int> d_j(1), d_s(1);  // pre-allocated for pick_move_rank_gpu
    thrust::device_vector<int> d_bad(1);        // for detecting kernel out-of-bounds

    // Type ids (device 0-based)
    const int t_nh_n = p.NH_N_type - 1;
    const int t_nh_p = p.NH_P_type - 1;
    const int t_w    = p.W_type    - 1;
    const int t_cl   = p.Cl_type   - 1;

    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    long long accum_nprot = 0;
    long long accum_steps = 0;

    for (int step = 0; step < Amax; ++step){

        // 1) Compute Kx = |M(x)| on GPU
        long long Kx = compute_K_and_offsets(d_nh_sites, d_neigh_move, d_atoms.type,
                                             sys.natoms,
                                             t_nh_n, t_nh_p, t_w, t_cl,
                                             d_counts, d_off);

        if (Kx <= 0){
            mc_report() << "[cph] |M(x)|=0, stop at step " << step << "\n";
            break;
        }

        // 2) Sample rank r uniformly on host
        std::uniform_int_distribution<long long> urank(0, Kx - 1);
        long long r = urank(rng);

        // 3) Pick (i,j,s) with GPU assistance (throws on failure - no continue)
        int i, j, s;
        pick_move_rank_gpu(r, d_nh_sites, d_off, d_neigh_move, d_atoms.type,
                           sys.natoms,
                           t_nh_n, t_nh_p, t_w, t_cl,
                           d_j, d_s,
                           i, j, s);

        // 4) Save old state from host mirror (avoid implicit D2H per element)
        int old_ti_dev = sys.atoms[i].type - 1;
        int old_tj_dev = sys.atoms[j].type - 1;
        real old_qi    = sys.atoms[i].q;
        real old_qj    = sys.atoms[j].q;

        // 5) Apply trial move on device
        int* bad_ptr = thrust::raw_pointer_cast(d_bad.data());
        CUDA_CHECK(cudaMemset(bad_ptr, 0, sizeof(int)));
        k_apply_move<<<1,1>>>(i, j, s, sys.natoms,
                              thrust::raw_pointer_cast(d_atoms.type.data()),
                              thrust::raw_pointer_cast(d_atoms.q.data()),
                              t_nh_n, t_nh_p, t_w, t_cl,
                              bad_ptr);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
        {
            int h_bad = 0;
            CUDA_CHECK(cudaMemcpy(&h_bad, bad_ptr, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_bad != 0){
                throw std::runtime_error("k_apply_move: out-of-bounds i=" + std::to_string(i) + " j=" + std::to_string(j));
            }
        }

        // 6) Energy trial
        real E_trial = total_energy(d_atoms, d_params, d_neigh_energy, box, plan);
        real dE_sim  = E_trial - R.E;
        real dE_corr = dE_sim - (real)s * p.calibration_offset;

        // 7) Compute Ky = |M(y)| on GPU (after trial types)
        long long Ky = compute_K_and_offsets(d_nh_sites, d_neigh_move, d_atoms.type,
                                             sys.natoms,
                                             t_nh_n, t_nh_p, t_w, t_cl,
                                             d_counts, d_off);
        if (Ky <= 0){
            // Should not happen if move is reversible; indicates bug in neighbor CSR
            CUDA_CHECK(cudaMemset(bad_ptr, 0, sizeof(int)));
            k_restore_two<<<1,1>>>(i, old_ti_dev, old_qi, j, old_tj_dev, old_qj, sys.natoms,
                                   thrust::raw_pointer_cast(d_atoms.type.data()),
                                   thrust::raw_pointer_cast(d_atoms.q.data()),
                                   bad_ptr);
            CUDA_KERNEL_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            {
                int h_bad = 0;
                CUDA_CHECK(cudaMemcpy(&h_bad, bad_ptr, sizeof(int), cudaMemcpyDeviceToHost));
                if (h_bad != 0){
                    throw std::runtime_error("k_restore_two OOB while handling Ky<=0");
                }
            }
            throw std::runtime_error("[cph] Ky==0 after trial move. Check neighbor CSR symmetry.");
        }

        // 8) MH accept (log-space, NO clamp)
        double log_alpha =
              (double)(-p.beta * dE_corr)
            + (double)s * (double)mu_over_kT
            + std::log((double)Kx)
            - std::log((double)Ky);

        double u = uni01(rng);
        if (u <= 0.0) u = std::nextafter(0.0, 1.0);
        double logu = std::log(u);

        bool accept = (log_alpha >= 0.0) || (logu < log_alpha);

        ++R.attempted;

        if (accept){
            ++R.accepted;
            if (s == +1) ++R.accepted_forward; else ++R.accepted_backward;

            R.E = E_trial;
            R.Nprot += s;

            // Commit to host mirror (1-based type)
            if (s == +1){
                sys.atoms[i].type = p.NH_P_type; sys.atoms[i].q = +1.0;
                sys.atoms[j].type = p.Cl_type;   sys.atoms[j].q = -1.0;
            } else {
                sys.atoms[i].type = p.NH_N_type; sys.atoms[i].q = 0.0;
                sys.atoms[j].type = p.W_type;    sys.atoms[j].q = 0.0;
            }
        } else {
            // Restore device only (host mirror unchanged)
            CUDA_CHECK(cudaMemset(bad_ptr, 0, sizeof(int)));
            k_restore_two<<<1,1>>>(i, old_ti_dev, old_qi, j, old_tj_dev, old_qj, sys.natoms,
                                   thrust::raw_pointer_cast(d_atoms.type.data()),
                                   thrust::raw_pointer_cast(d_atoms.q.data()),
                                   bad_ptr);
            CUDA_KERNEL_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            {
                int h_bad = 0;
                CUDA_CHECK(cudaMemcpy(&h_bad, bad_ptr, sizeof(int), cudaMemcpyDeviceToHost));
                if (h_bad != 0){
                    throw std::runtime_error("k_restore_two: out-of-bounds i=" + std::to_string(i) + " j=" + std::to_string(j));
                }
            }
        }

        accum_nprot += (long long)R.Nprot;
        ++accum_steps;

        if (R.attempted % std::max(1, p.energy_interval) == 0){
            double inst_ratio = (R.Nh > 0) ? (double)R.Nprot / (double)R.Nh : 0.0;
            double avg_ratio  = (accum_steps > 0 && R.Nh > 0)
                              ? (double)accum_nprot / (double)(accum_steps * R.Nh)
                              : 0.0;
            real acc = (R.attempted > 0) ? (real)R.accepted / (real)R.attempted : (real)0;

            mc_report() << std::setprecision(6)
                        << "[cph] steps=" << R.attempted
                        << " E=" << R.E
                        << " |M|=" << Kx
                        << " Inst_f=" << inst_ratio
                        << " Avg_f=" << avg_ratio
                        << " acc=" << acc
                        << " fwd=" << R.accepted_forward
                        << " bwd=" << R.accepted_backward
                        << std::endl;
        }
    }

    double final_avg_ratio = (accum_steps > 0 && R.Nh > 0)
                           ? (double)accum_nprot / (double)(accum_steps * R.Nh)
                           : 0.0;

    mc_report() << "\n============================================\n";
    mc_report() << " CpHMD COMPLETED (Exact MH, proposal on GPU)\n";
    mc_report() << " Calibration Offset used: " << p.calibration_offset << "\n";
    mc_report() << " Total Steps Sampled:     " << accum_steps << "\n";
    mc_report() << " Final Avg Prot Ratio:    " << final_avg_ratio << "\n";
    mc_report() << "============================================\n" << std::endl;

    double snapshot_state = (R.Nh > 0) ? (double)R.Nprot / (double)R.Nh : 0.0;
    mc_report() << "[SNAPSHOT] Nprot=" << R.Nprot << " Nh=" << R.Nh 
                << " Final_State=" << snapshot_state << std::endl;

    return R;
}
