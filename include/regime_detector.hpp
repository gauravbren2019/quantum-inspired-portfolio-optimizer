#pragma once
#include <Eigen/Dense>
#include <deque>
#include <cmath>

// two-state realized vol regime switch, keeps the quadratic form PSD
// under negative-return crash conditions instead of flipping the Hamiltonian upside down
class RegimeDetector {
public:
    explicit RegimeDetector(size_t window = 200) : window_(window) {}

    void update(double ret) {
        hist_.push_back(ret);
        if (hist_.size() > window_) hist_.pop_front();
    }

    bool is_stressed() const {
        if (hist_.size() < window_) return false;
        double mean = 0; for (double r : hist_) mean += r; mean /= hist_.size();
        double var = 0; for (double r : hist_) var += (r - mean) * (r - mean);
        var /= hist_.size();
        double realized_vol = std::sqrt(var * 252.0 * 390.0); // annualized from 1-min bars
        return realized_vol > stress_threshold_;
    }

    // forces Sigma to stay PSD, retries with ridge boost on Cholesky failure
    static Eigen::MatrixXd enforce_psd(Eigen::MatrixXd sigma) {
        for (int attempt = 0; attempt < 5; attempt++) {
            Eigen::LLT<Eigen::MatrixXd> llt(sigma);
            if (llt.info() == Eigen::Success) return sigma;
            sigma += 1e-6 * std::pow(10, attempt) * Eigen::MatrixXd::Identity(sigma.rows(), sigma.rows());
        }
        return sigma;
    }

private:
    std::deque<double> hist_;
    size_t window_;
    double stress_threshold_ = 0.45;
};
