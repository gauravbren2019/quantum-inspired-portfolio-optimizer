#pragma once
#include <Eigen/Dense>

// penalty and covariance tracking that actually adapts tick to tick
// instead of decaying against a static quarter-old calibration
class EWMACalibrator {
public:
    EWMACalibrator(int N, double lambda_decay = 0.97)
        : decay_(lambda_decay), sigma_(Eigen::MatrixXd::Identity(N, N) * 1e-4),
          mu_(Eigen::VectorXd::Zero(N)), trace0_(N * 1e-4), M0_(1.0) {}

    void update(const Eigen::VectorXd& ret) {
        mu_ = decay_ * mu_ + (1 - decay_) * ret;
        Eigen::VectorXd centered = ret - mu_;
        sigma_ = decay_ * sigma_ + (1 - decay_) * (centered * centered.transpose());
    }

    double adaptive_penalty() const { return M0_ * (sigma_.trace() / trace0_); }
    const Eigen::MatrixXd& sigma() const { return sigma_; }
    const Eigen::VectorXd& mu() const { return mu_; }

private:
    double decay_;
    Eigen::MatrixXd sigma_;
    Eigen::VectorXd mu_;
    double trace0_, M0_;
};
