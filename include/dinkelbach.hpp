#pragma once
#include <Eigen/Dense>
#include <limits>
#include "bifurcation_solver.hpp"
#include "factor_model.hpp"

// Dinkelbach fractional programming with a warm-started solver
// one persistent oscillator state is reused across lambda updates instead of
// re-randomizing and re-ramping from zero every iteration
class DinkelbachSolver {
public:
    int iters_used = 0;

    DinkelbachSolver(SolverConfig cfg, double tol = 1e-3, int max_iter = 6)
        : cfg_(cfg), tol_(tol), max_iter_(max_iter) {}

    std::vector<double> maximize_sharpe(const Eigen::VectorXd& mu,
                                         const FactorModel& factor_model,
                                         const Eigen::VectorXd& admm_bias) {
        const int N = static_cast<int>(mu.size());
        BifurcationSolver solver(cfg_, 42);
        Eigen::VectorXd h(N), sigma_w(N), w(N), best_w = Eigen::VectorXd::Zero(N);

        double lambda = 0.1;
        double best_sharpe = -std::numeric_limits<double>::infinity();
        iters_used = 0;

        for (int k = 0; k < max_iter_; k++) {
            h = lambda * mu - admm_bias;

            int steps = (k == 0) ? cfg_.steps : std::max(20, cfg_.steps / 4);
            double p_begin = (k == 0) ? 0.0 : 0.6;
            solver.run(h, factor_model, steps, p_begin);

            auto w_raw = solver.readout();
            for (int i = 0; i < N; i++) w(i) = w_raw[i];

            factor_model.apply_into(w, sigma_w);
            double numerator = w.dot(mu);
            double denom = std::sqrt(std::max(w.dot(sigma_w), 1e-12));
            double sharpe = numerator / denom;

            if (sharpe > best_sharpe) {
                best_sharpe = sharpe;
                best_w = w;
            }

            double lambda_next = numerator / denom;
            iters_used = k + 1;
            if (std::abs(lambda_next - lambda) < tol_) { lambda = lambda_next; break; }
            lambda = lambda_next;
        }

        return std::vector<double>(best_w.data(), best_w.data() + N);
    }

private:
    SolverConfig cfg_;
    double tol_;
    int max_iter_;
};
