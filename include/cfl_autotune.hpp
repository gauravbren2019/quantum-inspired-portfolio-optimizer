#pragma once
#include <Eigen/Dense>
#include <cmath>

// self-tunes the solver's time step off the live spectral radius of J
// no offline grid search, no human in the loop
class CFLAutotune {
public:
    static double spectral_radius(const FactorModel& fm, int N, int iters = 30) {
        Eigen::VectorXd v = Eigen::VectorXd::Random(N);
        v.normalize();
        double lambda = 0;
        for (int i = 0; i < iters; i++) {
            Eigen::VectorXd Jv = fm.apply(v);
            lambda = Jv.norm();
            if (lambda < 1e-12) break;
            v = Jv / lambda;
        }
        return lambda;
    }

    static double stable_dt(double delta0, double xi0, double j_spectral_radius, double safety = 0.8) {
        double bound = 2.0 / std::sqrt(delta0 * delta0 + xi0 * j_spectral_radius + 1e-9);
        return safety * bound;
    }
};
