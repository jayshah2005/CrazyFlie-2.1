// Gradient aggregators for multi-objective TD3 (L2F_MOO): combination weights w from the
// M x M Gram matrix G = J J^T of the per-objective gradients (rows of J), so that the actor
// step uses J^T w. Plain C++ (no rl_tools), double precision; checked against TorchJD by
// l2f/sim/check_aggregators.py.
//
// The preference vector is all ones (TorchJD's default is all 1/M; every aggregator here is
// linear in it, so the weights are exactly M times TorchJD's default). With ones, gradients
// that do not conflict are simply added, as in ordinary TD3 ("sum").
//
//   sum          w = 1
//   upgrad       UPGrad (Quinton & Rey, arXiv 2406.16232): each gradient projected onto the dual
//                cone of all of them, then added. TorchJD defaults: G normalised by its trace
//                (zero if trace < 1e-4), plus 1e-4 I; one QP min v^T G v s.t. v >= e_i per row.
//   aligned_mtl  Aligned-MTL (Senushkin et al., CVPR 2023): w = B 1 with
//                B = sqrt(lambda_min) V Lambda^{-1/2} V^T over the eigenpairs above
//                lambda_max * M * eps (TorchJD's default scale_mode "min").
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace learning_to_fly::moo::agg {
    struct sum {};
    struct upgrad {};
    struct aligned_mtl {};

    // Symmetric eigendecomposition of the leading n x n block (n <= M) by cyclic Jacobi:
    // A = V diag(lambda) V^T, eigenvalues in decreasing order, eigenvectors in the columns of V.
    template <int M>
    void eigh(const double A_in[M][M], double lambda[M], double V[M][M], int n = M) {
        double A[M][M];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                A[i][j] = 0.5 * (A_in[i][j] + A_in[j][i]);
                V[i][j] = i == j;
            }
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0, total = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    total += A[i][j] * A[i][j];
                    if (i != j) off += A[i][j] * A[i][j];
                }
            if (off <= 1e-32 * total || off == 0) break;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++) {
                    if (A[p][q] == 0) continue;
                    double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
                    double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                    double c = 1 / std::sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < n; k++) {  // A <- A J (columns p, q)
                        double akp = A[k][p], akq = A[k][q];
                        A[k][p] = c * akp - s * akq;
                        A[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {  // A <- J^T A (rows p, q)
                        double apk = A[p][k], aqk = A[q][k];
                        A[p][k] = c * apk - s * aqk;
                        A[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = c * vkp - s * vkq;
                        V[k][q] = s * vkp + c * vkq;
                    }
                }
        }
        int order[M];
        for (int i = 0; i < n; i++) order[i] = i;
        std::sort(order, order + n, [&](int a, int b) { return A[a][a] > A[b][b]; });
        double Vs[M][M];
        for (int j = 0; j < n; j++) {
            lambda[j] = A[order[j]][order[j]];
            for (int i = 0; i < n; i++) Vs[i][j] = V[i][order[j]];
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) V[i][j] = Vs[i][j];
    }

    // Solves A x = b for the n x n leading block given by idx (Gaussian elimination, partial
    // pivoting). A is M x M; idx selects n rows/columns.
    template <int M>
    void solve_sub(const double A[M][M], const int* idx, int n, double* b) {
        double a[M][M + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = A[idx[i]][idx[j]];
            a[i][n] = b[i];
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++)
                if (std::abs(a[r][c]) > std::abs(a[piv][c])) piv = r;
            for (int j = 0; j <= n; j++) std::swap(a[c][j], a[piv][j]);
            for (int r = c + 1; r < n; r++) {
                double f = a[r][c] / a[c][c];
                for (int j = c; j <= n; j++) a[r][j] -= f * a[c][j];
            }
        }
        for (int r = n - 1; r >= 0; r--) {
            double s = a[r][n];
            for (int j = r + 1; j < n; j++) s -= a[r][j] * b[j];
            b[r] = s / a[r][r];
        }
    }

    // min_v 0.5 v^T Q v  subject to  v >= u   (Q positive definite), primal active-set method
    // started from v = u with every bound active. Returns the number of iterations (-1: no
    // convergence within the cap, v is then the last feasible iterate).
    template <int M>
    int box_qp(const double Q[M][M], const double u[M], double v[M]) {
        bool active[M];
        for (int i = 0; i < M; i++) { v[i] = u[i]; active[i] = true; }
        for (int it = 1; it <= 200; it++) {
            int free_idx[M], nf = 0;
            for (int i = 0; i < M; i++)
                if (!active[i]) free_idx[nf++] = i;
            // minimiser with the active bounds held at u: Q_FF p_F = -Q_FA u_A
            double p[M];
            for (int i = 0; i < M; i++) p[i] = active[i] ? u[i] : 0;
            if (nf > 0) {
                double rhs[M];
                for (int a = 0; a < nf; a++) {
                    double s = 0;
                    for (int j = 0; j < M; j++)
                        if (active[j]) s -= Q[free_idx[a]][j] * u[j];
                    rhs[a] = s;
                }
                solve_sub<M>(Q, free_idx, nf, rhs);
                for (int a = 0; a < nf; a++) p[free_idx[a]] = rhs[a];
            }
            // step towards p until a free variable hits its bound
            double alpha = 1;
            int block = -1;
            for (int a = 0; a < nf; a++) {
                int i = free_idx[a];
                if (p[i] < u[i] && p[i] < v[i]) {
                    double t = (v[i] - u[i]) / (v[i] - p[i]);
                    if (t < alpha) { alpha = t; block = i; }
                }
            }
            for (int i = 0; i < M; i++) v[i] += alpha * (p[i] - v[i]);
            if (block >= 0) {
                v[block] = u[block];
                active[block] = true;
                continue;
            }
            // at the subproblem minimiser: multipliers mu = (Q v) on the active bounds must be >= 0
            double mu_min = 0, scale = 0;
            int drop = -1;
            for (int i = 0; i < M; i++) {
                double g = 0;
                for (int j = 0; j < M; j++) g += Q[i][j] * v[j];
                scale = std::max(scale, std::abs(g));
                if (active[i] && g < mu_min) { mu_min = g; drop = i; }
            }
            if (drop < 0 || mu_min >= -1e-14 * scale) return it;
            active[drop] = false;
        }
        return -1;
    }

    template <int M>
    void weights(sum, const double[M][M], double w[M]) {
        for (int k = 0; k < M; k++) w[k] = 1;
    }

    template <int M>
    void weights(upgrad, const double G[M][M], double w[M]) {
        constexpr double norm_eps = 1e-4, reg_eps = 1e-4;
        double trace = 0;
        for (int i = 0; i < M; i++) trace += G[i][i];
        double Q[M][M];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++) Q[i][j] = (trace < norm_eps ? 0.0 : G[i][j] / trace) + (i == j) * reg_eps;
        for (int k = 0; k < M; k++) w[k] = 0;
        for (int i = 0; i < M; i++) {
            double u[M], v[M];
            for (int k = 0; k < M; k++) u[k] = k == i;
            box_qp<M>(Q, u, v);
            for (int k = 0; k < M; k++) w[k] += v[k];
        }
    }

    template <int M>
    void weights(aligned_mtl, const double G[M][M], double w[M]) {
        double lambda[M], V[M][M];
        eigh<M>(G, lambda, V);
        const double tol = lambda[0] * M * std::numeric_limits<double>::epsilon();
        int rank = 0;
        while (rank < M && lambda[rank] > tol) rank++;
        if (rank == 0) {  // B = identity
            for (int k = 0; k < M; k++) w[k] = 1;
            return;
        }
        // w = sqrt(lambda_min) V Lambda^{-1/2} V^T 1
        const double s = std::sqrt(lambda[rank - 1]);
        for (int k = 0; k < M; k++) w[k] = 0;
        for (int j = 0; j < rank; j++) {
            double vt1 = 0;
            for (int k = 0; k < M; k++) vt1 += V[k][j];
            const double c = s / std::sqrt(lambda[j]) * vt1;
            for (int k = 0; k < M; k++) w[k] += c * V[k][j];
        }
    }
}
