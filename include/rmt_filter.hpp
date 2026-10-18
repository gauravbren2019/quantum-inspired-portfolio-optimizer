#pragma once
#include <Eigen/Dense>
#include <cmath>

// Marcenko-Pastur eigenvalue cleaning, stops HHL-style truncation disasters
// before the covariance ever touches the solver
class RMTFilter {
public:
    static Eigen::MatrixXd clean(const Eigen::MatrixXd& sample_cov, int N, int T) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sample_cov);
        Eigen::VectorXd eigvals = es.eigenvalues();
        Eigen::MatrixXd eigvecs = es.eigenvectors();

        double avg_var = sample_cov.trace() / N;
        double q = static_cast<double>(N) / T;
        double lambda_plus = avg_var * std::pow(1.0 + std::sqrt(q), 2);

        double noise_floor = 0.0;
        int n_noise = 0;
        for (int i = 0; i < N; i++) {
            if (eigvals(i) < lambda_plus) { noise_floor += eigvals(i); n_noise++; }
        }
        noise_floor = (n_noise > 0) ? noise_floor / n_noise : 1e-8;

        Eigen::VectorXd cleaned = eigvals;
        for (int i = 0; i < N; i++)
            if (cleaned(i) < lambda_plus) cleaned(i) = noise_floor;

        Eigen::MatrixXd sigma_clean = eigvecs * cleaned.asDiagonal() * eigvecs.transpose();
        // small ridge term so Cholesky never fails downstream
        sigma_clean += 1e-8 * Eigen::MatrixXd::Identity(N, N);
        return sigma_clean;
    }
};
