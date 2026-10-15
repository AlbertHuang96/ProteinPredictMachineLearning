#include "ppml/DistogramBins.h"
#include <cstring>
#include <cmath>

namespace ppml {

// ============================================================================
// 辅助: 从 (N, 3, 3) 坐标张量中提取原子坐标
// 布局: coords[i][atom][xyz] → data[i*9 + atom*3 + xyz]
//       atom 0=N, 1=CA, 2=C
// ============================================================================
#define COORD_ATOM(data, i, atom, c) ((data)[(i) * 9 + (atom) * 3 + (c)])

// ============================================================================
// compute_distance_onehot
// ============================================================================

void compute_distance_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* D_onehot)
{
    const int N = static_cast<int>(coords.shape().dims[0]);
    const int total_bins = DIST_N_BINS + 1; // 61 (60 + overflow)

    std::memset(D_onehot, 0, N * N * total_bins * sizeof(float));

    const float* cdata = coords.data();

    for (int l = 0; l < N; ++l) {
        if (seq_mask[l] < 0.5f) continue;

        // pseudo-Cβ for residue l
        float cbl_x, cbl_y, cbl_z;
        pseudo_cb(
            COORD_ATOM(cdata, l, 0, 0), COORD_ATOM(cdata, l, 0, 1), COORD_ATOM(cdata, l, 0, 2),
            COORD_ATOM(cdata, l, 1, 0), COORD_ATOM(cdata, l, 1, 1), COORD_ATOM(cdata, l, 1, 2),
            COORD_ATOM(cdata, l, 2, 0), COORD_ATOM(cdata, l, 2, 1), COORD_ATOM(cdata, l, 2, 2),
            cbl_x, cbl_y, cbl_z
        );

        for (int lp = 0; lp < N; ++lp) {
            if (seq_mask[lp] < 0.5f) continue;

            // pseudo-Cβ for residue l'
            float cblp_x, cblp_y, cblp_z;
            pseudo_cb(
                COORD_ATOM(cdata, lp, 0, 0), COORD_ATOM(cdata, lp, 0, 1), COORD_ATOM(cdata, lp, 0, 2),
                COORD_ATOM(cdata, lp, 1, 0), COORD_ATOM(cdata, lp, 1, 1), COORD_ATOM(cdata, lp, 1, 2),
                COORD_ATOM(cdata, lp, 2, 0), COORD_ATOM(cdata, lp, 2, 1), COORD_ATOM(cdata, lp, 2, 2),
                cblp_x, cblp_y, cblp_z
            );

            // distance
            float dx = cbl_x - cblp_x;
            float dy = cbl_y - cblp_y;
            float dz = cbl_z - cblp_z;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            // clamp to 18.5
            dist = std::min(dist, DIST_CLAMP);

            // binning
            int bin = distance_to_bin(dist);

            // write one-hot: D_onehot[l*N*61 + lp*61 + bin]
            int base = l * N * total_bins + lp * total_bins;
            D_onehot[base + bin] = 1.0f;
        }
    }
}

// ============================================================================
// compute_omega_onehot  — Ω[l][l'] = Dihedral(CA_l, CB_l, CA_l', CB_l')
// ============================================================================

void compute_omega_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* omega_onehot)
{
    const int N = static_cast<int>(coords.shape().dims[0]);

    std::memset(omega_onehot, 0, N * N * OMEGA_N_BINS * sizeof(float));

    const float* cdata = coords.data();

    for (int l = 0; l < N; ++l) {
        if (seq_mask[l] < 0.5f) continue;

        // CA_l, CB_l
        float cax_l = COORD_ATOM(cdata, l, 1, 0);
        float cay_l = COORD_ATOM(cdata, l, 1, 1);
        float caz_l = COORD_ATOM(cdata, l, 1, 2);

        float cbl_x, cbl_y, cbl_z;
        pseudo_cb(
            COORD_ATOM(cdata, l, 0, 0), COORD_ATOM(cdata, l, 0, 1), COORD_ATOM(cdata, l, 0, 2),
            cax_l, cay_l, caz_l,
            COORD_ATOM(cdata, l, 2, 0), COORD_ATOM(cdata, l, 2, 1), COORD_ATOM(cdata, l, 2, 2),
            cbl_x, cbl_y, cbl_z
        );

        for (int lp = 0; lp < N; ++lp) {
            if (seq_mask[lp] < 0.5f) continue;

            // CA_l', CB_l'
            float cax_lp = COORD_ATOM(cdata, lp, 1, 0);
            float cay_lp = COORD_ATOM(cdata, lp, 1, 1);
            float caz_lp = COORD_ATOM(cdata, lp, 1, 2);

            float cblp_x, cblp_y, cblp_z;
            pseudo_cb(
                COORD_ATOM(cdata, lp, 0, 0), COORD_ATOM(cdata, lp, 0, 1), COORD_ATOM(cdata, lp, 0, 2),
                cax_lp, cay_lp, caz_lp,
                COORD_ATOM(cdata, lp, 2, 0), COORD_ATOM(cdata, lp, 2, 1), COORD_ATOM(cdata, lp, 2, 2),
                cblp_x, cblp_y, cblp_z
            );

            // Ω = Dihedral(CA_l, CB_l, CA_l', CB_l')
            float angle = dihedral_angle(
                cax_l,  cay_l,  caz_l,    // CA_l
                cbl_x,  cbl_y,  cbl_z,    // CB_l
                cax_lp, cay_lp, caz_lp,    // CA_l'
                cblp_x, cblp_y, cblp_z     // CB_l'
            );

            int bin = dihedral_to_bin(angle);
            int base = l * N * OMEGA_N_BINS + lp * OMEGA_N_BINS;
            omega_onehot[base + bin] = 1.0f;
        }
    }
}

// ============================================================================
// compute_theta_onehot  — Θ[l][l'] = Dihedral(N_l, CA_l, CB_l, CB_l')
// ============================================================================

void compute_theta_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* theta_onehot)
{
    const int N = static_cast<int>(coords.shape().dims[0]);

    std::memset(theta_onehot, 0, N * N * THETA_N_BINS * sizeof(float));

    const float* cdata = coords.data();

    for (int l = 0; l < N; ++l) {
        if (seq_mask[l] < 0.5f) continue;

        // N_l, CA_l, CB_l
        float nx_l  = COORD_ATOM(cdata, l, 0, 0);
        float ny_l  = COORD_ATOM(cdata, l, 0, 1);
        float nz_l  = COORD_ATOM(cdata, l, 0, 2);
        float cax_l = COORD_ATOM(cdata, l, 1, 0);
        float cay_l = COORD_ATOM(cdata, l, 1, 1);
        float caz_l = COORD_ATOM(cdata, l, 1, 2);

        float cbl_x, cbl_y, cbl_z;
        pseudo_cb(
            nx_l, ny_l, nz_l,
            cax_l, cay_l, caz_l,
            COORD_ATOM(cdata, l, 2, 0), COORD_ATOM(cdata, l, 2, 1), COORD_ATOM(cdata, l, 2, 2),
            cbl_x, cbl_y, cbl_z
        );

        for (int lp = 0; lp < N; ++lp) {
            if (seq_mask[lp] < 0.5f) continue;

            // CB_l'
            float cax_lp = COORD_ATOM(cdata, lp, 1, 0);
            float cay_lp = COORD_ATOM(cdata, lp, 1, 1);
            float caz_lp = COORD_ATOM(cdata, lp, 1, 2);

            float cblp_x, cblp_y, cblp_z;
            pseudo_cb(
                COORD_ATOM(cdata, lp, 0, 0), COORD_ATOM(cdata, lp, 0, 1), COORD_ATOM(cdata, lp, 0, 2),
                cax_lp, cay_lp, caz_lp,
                COORD_ATOM(cdata, lp, 2, 0), COORD_ATOM(cdata, lp, 2, 1), COORD_ATOM(cdata, lp, 2, 2),
                cblp_x, cblp_y, cblp_z
            );

            // Θ = Dihedral(N_l, CA_l, CB_l, CB_l')
            float angle = dihedral_angle(
                nx_l,   ny_l,   nz_l,     // N_l
                cax_l,  cay_l,  caz_l,    // CA_l
                cbl_x,  cbl_y,  cbl_z,    // CB_l
                cblp_x, cblp_y, cblp_z    // CB_l'
            );

            int bin = dihedral_to_bin(angle);
            int base = l * N * THETA_N_BINS + lp * THETA_N_BINS;
            theta_onehot[base + bin] = 1.0f;
        }
    }
}

// ============================================================================
// compute_phi_onehot  — Φ[l][l'] = Planar(CA_l, CB_l, CB_l')
// ============================================================================

void compute_phi_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* phi_onehot)
{
    const int N = static_cast<int>(coords.shape().dims[0]);

    std::memset(phi_onehot, 0, N * N * PHI_N_BINS * sizeof(float));

    const float* cdata = coords.data();

    for (int l = 0; l < N; ++l) {
        if (seq_mask[l] < 0.5f) continue;

        // CA_l, CB_l
        float cax_l = COORD_ATOM(cdata, l, 1, 0);
        float cay_l = COORD_ATOM(cdata, l, 1, 1);
        float caz_l = COORD_ATOM(cdata, l, 1, 2);

        float cbl_x, cbl_y, cbl_z;
        pseudo_cb(
            COORD_ATOM(cdata, l, 0, 0), COORD_ATOM(cdata, l, 0, 1), COORD_ATOM(cdata, l, 0, 2),
            cax_l, cay_l, caz_l,
            COORD_ATOM(cdata, l, 2, 0), COORD_ATOM(cdata, l, 2, 1), COORD_ATOM(cdata, l, 2, 2),
            cbl_x, cbl_y, cbl_z
        );

        for (int lp = 0; lp < N; ++lp) {
            if (seq_mask[lp] < 0.5f) continue;

            // CB_l'
            float cax_lp = COORD_ATOM(cdata, lp, 1, 0);
            float cay_lp = COORD_ATOM(cdata, lp, 1, 1);
            float caz_lp = COORD_ATOM(cdata, lp, 1, 2);

            float cblp_x, cblp_y, cblp_z;
            pseudo_cb(
                COORD_ATOM(cdata, lp, 0, 0), COORD_ATOM(cdata, lp, 0, 1), COORD_ATOM(cdata, lp, 0, 2),
                cax_lp, cay_lp, caz_lp,
                COORD_ATOM(cdata, lp, 2, 0), COORD_ATOM(cdata, lp, 2, 1), COORD_ATOM(cdata, lp, 2, 2),
                cblp_x, cblp_y, cblp_z
            );

            // Φ = Planar(CA_l, CB_l, CB_l')
            float angle = planar_angle(
                cax_l,  cay_l,  caz_l,    // CA_l (顶点)
                cbl_x,  cbl_y,  cbl_z,    // CB_l
                cblp_x, cblp_y, cblp_z    // CB_l'
            );

            int bin = planar_to_bin(angle);
            int base = l * N * PHI_N_BINS + lp * PHI_N_BINS;
            phi_onehot[base + bin] = 1.0f;
        }
    }
}

// ============================================================================
// compute_all_distogram_onehots — 一次性计算全部 4 个
// ============================================================================

void compute_all_distogram_onehots(
    const TensorF32& coords,
    const float* seq_mask,
    float* D_onehot,
    float* omega_onehot,
    float* theta_onehot,
    float* phi_onehot)
{
    compute_distance_onehot(coords, seq_mask, D_onehot);
    compute_omega_onehot(coords, seq_mask, omega_onehot);
    compute_theta_onehot(coords, seq_mask, theta_onehot);
    compute_phi_onehot(coords, seq_mask, phi_onehot);
}

} // namespace ppml
