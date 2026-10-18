#pragma once
#include <Eigen/Dense>

// keeps the "pick exactly M assets" constraint out of the main Hamiltonian
// so it never drowns out real covariance signal with a giant penalty term
class ADMMCardinality {
public:
    ADMMCardinality(int target_M, double rho = 0.5) : M_(target_M), rho_(rho), lambda_(0.0) {}

    // returns a per-asset linear bias to add to h, not a quadratic penalty
    Eigen::VectorXd dual_bias(int N) const {
        return Eigen::VectorXd::Constant(N, lambda_);
    }

    void update(int active_count) {
        lambda_ += rho_ * (active_count - M_);
    }

    double lambda() const { return lambda_; }

private:
    int M_;
    double rho_;
    double lambda_;
};
