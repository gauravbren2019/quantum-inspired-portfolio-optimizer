#pragma once
#include <Eigen/Dense>

struct FactorModel {
    Eigen::MatrixXd B;
    Eigen::MatrixXd F;
    Eigen::VectorXd D;

    // scratch buffers, allocated once in fit(), reused forever after, no heap churn in the hot path
    mutable Eigen::VectorXd scratch_k_;
    mutable Eigen::VectorXd scratch_k2_;
    mutable Eigen::VectorXd scratch_n_;

    static FactorModel fit(const Eigen::MatrixXd& sigma_clean, int K) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sigma_clean);
        int N = sigma_clean.rows();
        Eigen::VectorXd vals = es.eigenvalues().tail(K).reverse();
        Eigen::MatrixXd vecs = es.eigenvectors().rightCols(K).rowwise().reverse();

        FactorModel fm;
        fm.F = vals.asDiagonal();
        fm.B = vecs;
        Eigen::MatrixXd approx = fm.B * fm.F * fm.B.transpose();
        fm.D = (sigma_clean.diagonal() - approx.diagonal()).cwiseMax(1e-6);

        fm.scratch_k_.resize(K);
        fm.scratch_k2_.resize(K);
        fm.scratch_n_.resize(N);
        return fm;
    }

    // returns a fresh vector, kept for code that still needs it (called rarely, outside hot loop)
    Eigen::VectorXd apply(const Eigen::VectorXd& x) const {
        Eigen::VectorXd out(x.size());
        apply_into(x, out);
        return out;
    }

    // writes result into out with zero heap allocation, call this inside any hot loop
    void apply_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
        scratch_k_.noalias() = B.transpose() * x;
        scratch_k2_.noalias() = F * scratch_k_;
        out.noalias() = B * scratch_k2_;
        out.array() += D.array() * x.array();
    }
};
