#pragma once
#include <vector>
#include "types.hpp"
#include "device_data.hpp"

struct HostNeigh {
    std::vector<std::vector<int>> nb; // symmetric neighbors on host
};

// Copy host CSR data to device-resident CSR (head/list).
DeviceCSR copy_csr_to_device(const CSR& host);

// Build half neighbor list (i<j) entirely on GPU using brute-force O(N^2/2),
// excluding 1-2 bonded pairs from device CSR adjacency. Returns device CSR.
DeviceCSR build_half_neighbors_gpu(const DeviceAtoms& atoms,
                                   const Box& box,
                                   const DeviceCSR& bonds12_dev,
                                   real rc);

// Copy device CSR back to host.
CSR copy_csr_from_device(const DeviceCSR& dev);

// Build symmetric host neighbors from half-CSR (i<j) data.
HostNeigh build_host_symmetric(const CSR& half, int N);
