// Test harness for aggregators.h: reads M x M Gram matrices (M*M numbers per line, row-major;
// M = first argument, 5, 8 or 9) from stdin and prints, per matrix and aggregator,
// "<index> <name> w_0 ... w_{M-1}".
// Built and compared against TorchJD by l2f/sim/check_aggregators.py.
#include <cstdio>
#include <cstdlib>
#include "aggregators.h"
#include "aggregators_pga.h"

using namespace learning_to_fly::moo;

template <int M, typename A>
void run(int index, const char* name, const double G[M][M]) {
    double w[M];
    agg::weights<M>(A{}, G, w);
    std::printf("%d %s", index, name);
    for (int k = 0; k < M; k++) std::printf(" %.17g", w[k]);
    std::printf("\n");
}

template <int M>
int run_all() {
    double G[M][M];
    for (int index = 0;; index++) {
        for (int i = 0; i < M * M; i++)
            if (std::scanf("%lf", &G[i / M][i % M]) != 1) return 0;
        run<M, agg::sum>(index, "sum", G);
        run<M, agg::upgrad>(index, "upgrad", G);
        run<M, agg::aligned_mtl>(index, "aligned_mtl", G);
        run<M, agg::svd_pga>(index, "svd_pga", G);
        run<M, agg::osnmf_pga>(index, "osnmf_pga", G);
    }
}

int main(int argc, char** argv) {  // argument: matrix size M (5, 8 or 9, default 8)
    int m = argc > 1 ? std::atoi(argv[1]) : 8;
    if (m == 5) return run_all<5>();
    if (m == 8) return run_all<8>();
    if (m == 9) return run_all<9>();
    std::fprintf(stderr, "unsupported size %d\n", m);
    return 1;
}
