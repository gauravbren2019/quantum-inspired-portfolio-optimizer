#pragma once
#include <Eigen/Dense>
#include <algorithm>

// SB has no formal convergence proof, so treat its output as a warm start
// and run a few deterministic projected gradient steps for a guaranteed local optimum
class PGDPolish {
public:
    static Eigen::VectorXd polish(Eigen::VectorXd w, const Eigen::VectorXd& mu,
                                   const FactorModel& fm, double lambda,
                                   double lr = 0.01, int iters = 5) {
        for (int it = 0; it < iters; it++) {
            Eigen::VectorXd grad = 2.0 * fm.apply(w) - lambda * mu;
            w -= lr * grad;
            double max_abs = w.cwiseAbs().maxCoeff();
            if (max_abs > 1.0) w /= max_abs; // project back into feasible box
        }
        return w;
    }
};
