// C++ port of the SVDPGA / OSNMFPGA weightings by R. Khosrowshahi
// (https://github.com/rkhosrowshahi/torchjd, branch feat/pga-aggregators,
// src/torchjd/aggregation/_pga.py; TorchJD is MIT-licensed), defaults: rank None,
// eigenvalue_tol 1e-10, max_iters 30, tol 1e-9, roundoff_scale 16.
// Selected with AGG=svd_pga / AGG=osnmf_pga.
#pragma once
#include "aggregators.h"

namespace learning_to_fly::moo::agg {
    struct svd_pga {};
    struct osnmf_pga {};

    namespace pga {
        constexpr double eigenvalue_tol = 1e-10;
        constexpr int max_iters = 30;
        constexpr double tol = 1e-9;
        constexpr double roundoff_scale = 16;

        // _initial_basis: basis weights B (M x t, oriented, unit principal directions G B) and
        // the coordinates C (n_pos x M) of the gradients in the principal basis. Returns t.
        template <int M>
        int initial_basis(const double gram[M][M], double B[M][M], double C[M][M], int& n_pos) {
            double lambda[M], V[M][M];
            eigh<M>(gram, lambda, V);
            for (int i = 0; i < M; i++) lambda[i] = std::max(lambda[i], 0.0);  // clamp, order kept
            n_pos = 0;
            for (int i = 0; i < M; i++)
                if (lambda[i] > eigenvalue_tol * lambda[0]) n_pos++;
            const int t = n_pos;
            for (int j = 0; j < t; j++) {
                double pos = 0, neg = 0;
                for (int i = 0; i < M; i++) (V[i][j] > 0 ? pos : neg) += V[i][j] * V[i][j];
                const double sign = neg > pos ? -1.0 : 1.0;
                for (int i = 0; i < M; i++) B[i][j] = sign * V[i][j] / std::sqrt(lambda[j]);
            }
            for (int j = 0; j < n_pos; j++)
                for (int i = 0; i < M; i++) C[j][i] = std::sqrt(lambda[j]) * V[i][j];
            return t;
        }

        // _normalize: symmetric part divided by its largest diagonal entry.
        template <int M>
        double normalize(const double G[M][M], double gram[M][M]) {
            double scale = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++) gram[i][j] = 0.5 * (G[i][j] + G[j][i]);
            for (int i = 0; i < M; i++) scale = std::max(scale, gram[i][i]);
            if (scale <= 0) return 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++) gram[i][j] /= scale;
            return scale;
        }

        // 0.5 * ||J - W H||^2 for W = J B (t columns), H t x M, from the Gramian.
        template <int M>
        double gram_loss(const double gram[M][M], const double B[M][M], const double H[M][M], int t) {
            double GB[M][M] = {};  // M x t
            for (int i = 0; i < M; i++)
                for (int j = 0; j < t; j++)
                    for (int k = 0; k < M; k++) GB[i][j] += gram[i][k] * B[k][j];
            double BtGB[M][M] = {};  // t x t
            for (int a = 0; a < t; a++)
                for (int b = 0; b < t; b++)
                    for (int k = 0; k < M; k++) BtGB[a][b] += B[k][a] * GB[k][b];
            double trace = 0, cross = 0, quad = 0;
            for (int i = 0; i < M; i++) trace += gram[i][i];
            for (int a = 0; a < t; a++)
                for (int i = 0; i < M; i++) cross += H[a][i] * GB[i][a];
            for (int a = 0; a < t; a++)
                for (int i = 0; i < M; i++) {
                    double s = 0;
                    for (int b = 0; b < t; b++) s += BtGB[a][b] * H[b][i];
                    quad += H[a][i] * s;
                }
            return 0.5 * (trace - 2 * cross + quad);
        }

        // H = clamp(B^T gram, 0), t x M
        template <int M>
        void factor_h(const double gram[M][M], const double B[M][M], int t, double H[M][M]) {
            for (int a = 0; a < t; a++)
                for (int i = 0; i < M; i++) {
                    double s = 0;
                    for (int k = 0; k < M; k++) s += B[k][a] * gram[k][i];
                    H[a][i] = std::max(s, 0.0);
                }
        }

        // _gram_procrustes_step: B such that J B is the orthonormal basis closest to J H^T.
        template <int M>
        void procrustes(const double gram[M][M], const double H[M][M], const double C[M][M], int t, int n_pos,
                        double B[M][M]) {
            double S[M][M] = {};  // H gram H^T, t x t
            for (int a = 0; a < t; a++)
                for (int b = 0; b < t; b++) {
                    double s = 0;
                    for (int i = 0; i < M; i++)
                        for (int k = 0; k < M; k++) s += H[a][i] * gram[i][k] * H[b][k];
                    S[a][b] = s;
                }
            double e[M], E[M][M];
            eigh<M>(S, e, E, t);  // decreasing: smallest is e[t-1]
            if (e[t - 1] <= eigenvalue_tol * e[0]) {
                // fallback: (C / rowsumsq(C))^T @ polar(C H^T); polar via the eigenpairs of
                // A^T A, A = C H^T (n_pos x t); null directions completed orthonormally (their
                // choice is arbitrary, as in the SVD)
                double A[M][M] = {};
                for (int r = 0; r < n_pos; r++)
                    for (int a = 0; a < t; a++)
                        for (int i = 0; i < M; i++) A[r][a] += C[r][i] * H[a][i];
                double AtA[M][M] = {};
                for (int a = 0; a < t; a++)
                    for (int b = 0; b < t; b++)
                        for (int r = 0; r < n_pos; r++) AtA[a][b] += A[r][a] * A[r][b];
                double s2[M], W[M][M];
                eigh<M>(AtA, s2, W, t);
                double U[M][M] = {};  // n_pos x t, orthonormal columns
                for (int j = 0; j < t; j++) {
                    const double sj = std::sqrt(std::max(s2[j], 0.0));
                    bool ok = sj > 1e-12 * std::sqrt(std::max(s2[0], 0.0));
                    if (ok) {
                        for (int r = 0; r < n_pos; r++) {
                            double s = 0;
                            for (int a = 0; a < t; a++) s += A[r][a] * W[a][j];
                            U[r][j] = s / sj;
                        }
                    } else {
                        for (int cand = 0; cand < n_pos; cand++) {  // Gram-Schmidt on e_cand
                            double x[M] = {};
                            x[cand] = 1;
                            for (int q = 0; q < j; q++) {
                                double d = U[cand][q];
                                for (int r = 0; r < n_pos; r++) x[r] -= d * U[r][q];
                            }
                            double nrm = 0;
                            for (int r = 0; r < n_pos; r++) nrm += x[r] * x[r];
                            if (nrm > 1e-6) {
                                for (int r = 0; r < n_pos; r++) U[r][j] = x[r] / std::sqrt(nrm);
                                break;
                            }
                        }
                    }
                }
                double P[M][M] = {};  // polar = U W^T, n_pos x t
                for (int r = 0; r < n_pos; r++)
                    for (int a = 0; a < t; a++)
                        for (int j = 0; j < t; j++) P[r][a] += U[r][j] * W[a][j];
                for (int i = 0; i < M; i++)
                    for (int a = 0; a < t; a++) {
                        double s = 0;
                        for (int r = 0; r < n_pos; r++) {
                            double rs = 0;
                            for (int k = 0; k < M; k++) rs += C[r][k] * C[r][k];
                            s += C[r][i] / rs * P[r][a];
                        }
                        B[i][a] = s;
                    }
                return;
            }
            // B = H^T S^{-1/2}
            for (int i = 0; i < M; i++)
                for (int a = 0; a < t; a++) {
                    double s = 0;
                    for (int j = 0; j < t; j++) {
                        double hE = 0;
                        for (int b = 0; b < t; b++) hE += H[b][i] * E[b][j];
                        s += hE / std::sqrt(e[j]) * E[a][j];
                    }
                    B[i][a] = s;
                }
        }
    }

    template <int M>
    void weights(svd_pga, const double G[M][M], double w[M]) {
        double gram[M][M], B[M][M], C[M][M];
        int n_pos = 0;
        for (int k = 0; k < M; k++) w[k] = 0;
        const double scale = pga::normalize<M>(G, gram);
        if (scale <= 0) return;
        const int t = pga::initial_basis<M>(gram, B, C, n_pos);
        if (t == 0) return;
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < t; j++) w[k] += B[k][j];
            w[k] /= t * std::sqrt(scale);
        }
    }

    template <int M>
    void weights(osnmf_pga, const double G[M][M], double w[M]) {
        double gram[M][M], B[M][M], C[M][M], H[M][M];
        int n_pos = 0;
        for (int k = 0; k < M; k++) w[k] = 0;
        const double scale = pga::normalize<M>(G, gram);
        if (scale <= 0) return;
        const int t = pga::initial_basis<M>(gram, B, C, n_pos);
        if (t == 0) return;
        double energy = 0;
        for (int i = 0; i < M; i++) energy += gram[i][i];
        const double roundoff = pga::roundoff_scale * std::numeric_limits<double>::epsilon() * energy;
        pga::factor_h<M>(gram, B, t, H);
        double previous = pga::gram_loss<M>(gram, B, H, t);
        for (int it = 0; it < pga::max_iters; it++) {
            pga::procrustes<M>(gram, H, C, t, n_pos, B);
            const double loss = pga::gram_loss<M>(gram, B, H, t);
            pga::factor_h<M>(gram, B, t, H);
            if (std::abs(previous - loss) <= pga::tol * std::abs(previous) + roundoff) break;
            previous = loss;
        }
        for (int k = 0; k < M; k++) {
            for (int j = 0; j < t; j++) w[k] += B[k][j];
            w[k] /= t * std::sqrt(scale);
        }
    }
}
