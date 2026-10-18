#pragma once
#include <Eigen/Dense>
#include <vector>

class CoskewTensorCP {
public:
    CoskewTensorCP(int N, int R) : N_(N), R_(R) {
        U = Eigen::MatrixXd::Random(N, R) * 0.1;
        V = Eigen::MatrixXd::Random(N, R) * 0.1;
        W = Eigen::MatrixXd::Random(N, R) * 0.1;
        lambda = Eigen::VectorXd::Ones(R) * 0.01;
    }

    void fit(const std::vector<Eigen::VectorXd>& returns_window, int iters = 20) {
        int T = returns_window.size();
        Eigen::MatrixXd R3(T, N_);
        for (int t = 0; t < T; t++) R3.row(t) = returns_window[t].transpose();

        for (int it = 0; it < iters; it++) {
            Eigen::MatrixXd khatri = khatri_rao(V, W);
            U = (R3.transpose() * khatri_proxy(R3, khatri)) * pinv(khatri.transpose() * khatri);
            khatri = khatri_rao(U, W);
            V = (R3.transpose() * khatri_proxy(R3, khatri)) * pinv(khatri.transpose() * khatri);
            khatri = khatri_rao(U, V);
            W = (R3.transpose() * khatri_proxy(R3, khatri)) * pinv(khatri.transpose() * khatri);
        }
    }

    Eigen::VectorXd force(const Eigen::VectorXd& x) const {
        Eigen::VectorXd f = Eigen::VectorXd::Zero(N_);
        for (int r = 0; r < R_; r++) {
            double xv = x.dot(V.col(r));
            double xw = x.dot(W.col(r));
            f += lambda(r) * xv * xw * U.col(r);
        }
        return f;
    }

private:
    static Eigen::MatrixXd khatri_rao(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        int I = A.rows(), J = B.rows(), R = A.cols();
        Eigen::MatrixXd out(I * J, R);
        for (int r = 0; r < R; r++)
            for (int i = 0; i < I; i++)
                for (int j = 0; j < J; j++)
                    out(i * J + j, r) = A(i, r) * B(j, r);
        return out;
    }
    static Eigen::MatrixXd khatri_proxy(const Eigen::MatrixXd& R3, const Eigen::MatrixXd&) { return R3; }
    static Eigen::MatrixXd pinv(const Eigen::MatrixXd& M) {
        return M.completeOrthogonalDecomposition().pseudoInverse();
    }

    int N_, R_;
    Eigen::MatrixXd U, V, W;
    Eigen::VectorXd lambda;
};
