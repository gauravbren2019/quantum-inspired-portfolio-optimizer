#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <chrono>
#include "factor_model.hpp"
#include "rmt_filter.hpp"
#include "regime_detector.hpp"
#include "admm_cardinality.hpp"
#include "cfl_autotune.hpp"
#include "dinkelbach.hpp"
#include "pgd_polish.hpp"

// shared memory layout agreed with python/shared_mem_bridge.py
// [0]            : ready flag, set by python after writing mu/sigma
// [1]            : N (number of live symbols)
// [2 .. 2+N]     : mu vector
// [2+N .. 2+N+N*N] : sigma matrix, row major
// output region written back after [header + mu + sigma]
struct SharedLayout { double* base; int N; };

int main() {
    const char* shm_name = "/qisb_live";
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd < 0) { std::cerr << "shared memory not found, start python feed first\n"; return 1; }

    size_t region_size = 8 * 1024 * 1024;
    double* mem = static_cast<double*>(mmap(nullptr, region_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));

    ADMMCardinality admm(10); // target 10 live names out of the stream
    std::cout << "live engine warm and resident, waiting on tick updates\n";

    while (true) {
        if (mem[0] != 1.0) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }

        int N = static_cast<int>(mem[1]);
        Eigen::VectorXd mu(N);
        Eigen::MatrixXd sigma(N, N);
        for (int i = 0; i < N; i++) mu(i) = mem[2 + i];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                sigma(i, j) = mem[2 + N + i * N + j];

        auto t0 = std::chrono::high_resolution_clock::now();

        Eigen::MatrixXd sigma_clean = RMTFilter::clean(sigma, N, 500);
        sigma_clean = RegimeDetector::enforce_psd(sigma_clean);
        FactorModel fm = FactorModel::fit(sigma_clean, std::min(3, N));
        double spectral_radius = CFLAutotune::spectral_radius(fm, N);

        SolverConfig cfg;
        cfg.n_assets = N;
        cfg.bits_per_asset = 1;
        cfg.xi0 = 1.0 / std::max(spectral_radius, 1e-6);
        cfg.dt = CFLAutotune::stable_dt(cfg.delta0, cfg.xi0, spectral_radius);

        DinkelbachSolver dink(cfg);
        Eigen::VectorXd bias = admm.dual_bias(N);
        auto w_raw = dink.maximize_sharpe(mu, fm, bias);

        Eigen::VectorXd w(N);
        for (int i = 0; i < N; i++) w(i) = w_raw[i];
        Eigen::VectorXd w_final = PGDPolish::polish(w, mu, fm, 1.0);

        size_t out_offset = 2 + N + N * N;
        for (int i = 0; i < N; i++) mem[out_offset + i] = w_final(i);

        auto t1 = std::chrono::high_resolution_clock::now();
        mem[out_offset + N] = std::chrono::duration<double, std::micro>(t1 - t0).count(); // latency readback
        mem[0] = 2.0; // signal python output is ready
    }
    return 0;
}
