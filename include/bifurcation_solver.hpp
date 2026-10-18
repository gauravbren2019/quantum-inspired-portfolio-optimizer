#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include "factor_model.hpp"
#include "coskew_tensor.hpp"

struct SolverConfig {
    int n_assets;
    int bits_per_asset = 8;
    double lot_size = 1.0;
    double K = 1.0;
    double delta0 = 0.5;
    double xi0;
    double dt = 0.01;
    int steps = 150;
    double noise_sigma0 = 0.08;
};

class FastRng {
public:
    explicit FastRng(uint64_t seed) {
        s0_ = splitmix(seed);
        s1_ = splitmix(s0_);
    }

    double next_uniform() {
        uint64_t x = s0_;
        uint64_t y = s1_;
        s0_ = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        s1_ = x;
        uint64_t r = s0_ + s1_;
        return (r >> 11) * (1.0 / 9007199254740992.0);
    }

    double next_normal() {
        if (has_cached_) { has_cached_ = false; return cached_; }
        double u1 = std::max(next_uniform(), 1e-12);
        double u2 = next_uniform();
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 6.283185307179586 * u2;
        cached_ = r * std::sin(theta);
        has_cached_ = true;
        return r * std::cos(theta);
    }

private:
    static uint64_t splitmix(uint64_t x) {
        x += 0x9E3779B97f4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }
    uint64_t s0_, s1_;
    bool has_cached_ = false;
    double cached_ = 0.0;
};

// persistent oscillator state, reused across Dinkelbach iterations
// warm starting avoids re-seeding and re-running the full adiabatic ramp every call
class BifurcationSolver {
public:
    explicit BifurcationSolver(const SolverConfig& cfg, uint64_t seed = 12345)
        : cfg_(cfg), n_osc_(cfg.n_assets * cfg.bits_per_asset),
          x_(n_osc_, 0.0), y_(n_osc_, 0.0),
          bit_weights_(cfg.bits_per_asset), w_cache_(cfg.n_assets), field_(cfg.n_assets),
          rng_(seed) {
        for (auto& v : x_) v = (rng_.next_uniform() - 0.5) * 0.1;

        double p = 1.0;
        for (int b = 0; b < cfg.bits_per_asset; b++) {
            bit_weights_[b] = p * cfg.lot_size;
            p *= 2.0;
        }
    }

    // runs n_steps starting from pump level p_begin, continuing from current oscillator state
    // call this repeatedly with a shrinking step count once the state is already close to converged
    void run(const Eigen::VectorXd& h_linear,
             const FactorModel& factor_model,
             int n_steps,
             double p_begin,
             const CoskewTensorCP* coskew = nullptr) {
        const double noise_floor = 1e-5;

        for (int step = 0; step < n_steps; step++) {
            double frac = static_cast<double>(step) / n_steps;
            double p_t = p_begin + (1.0 - p_begin) * frac;
            double sigma_t = cfg_.noise_sigma0 * (1.0 - p_t);
            bool inject_noise = sigma_t > noise_floor;

            refresh_weight_cache();
            factor_model.apply_into(w_cache_, field_);
            field_.noalias() = field_ - h_linear;
            if (coskew) field_.noalias() = field_ - coskew->force(w_cache_);

            for (int i = 0; i < cfg_.n_assets; i++) {
                double asset_field = field_(i);
                int base = i * cfg_.bits_per_asset;
                for (int b = 0; b < cfg_.bits_per_asset; b++) {
                    int k = base + b;
                    double local_field = bit_weights_[b] * asset_field;

                    double xi = x_[k];
                    double nonlinear = -(cfg_.K * xi * xi - p_t + cfg_.delta0) * xi;
                    double coupling = -cfg_.xi0 * local_field;
                    double stoch = inject_noise ? sigma_t * rng_.next_normal() : 0.0;

                    y_[k] += (nonlinear + coupling + stoch) * cfg_.dt;
                    xi += y_[k] * cfg_.dt;

                    if (xi > 1.0)  { xi = 1.0;  y_[k] = 0.0; }
                    if (xi < -1.0) { xi = -1.0; y_[k] = 0.0; }
                    x_[k] = xi;
                }
            }
        }
        refresh_weight_cache();
    }

    std::vector<double> readout() const {
        std::vector<double> out(cfg_.n_assets);
        for (int i = 0; i < cfg_.n_assets; i++) out[i] = w_cache_(i);
        return out;
    }

private:
    void refresh_weight_cache() {
        for (int i = 0; i < cfg_.n_assets; i++) {
            double acc = 0.0;
            int base = i * cfg_.bits_per_asset;
            for (int b = 0; b < cfg_.bits_per_asset; b++) {
                double xi = x_[base + b];
                double sign = (xi >= 0.0) ? 1.0 : -1.0;
                acc += sign * bit_weights_[b] * std::abs(xi);
            }
            w_cache_(i) = acc;
        }
    }

    SolverConfig cfg_;
    int n_osc_;
    std::vector<double> x_, y_;
    std::vector<double> bit_weights_;
    Eigen::VectorXd w_cache_;
    Eigen::VectorXd field_;
    FastRng rng_;
};
