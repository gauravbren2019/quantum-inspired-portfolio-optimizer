#pragma once
#include <cmath>

// real market impact, not a flat per-share fee
struct AlmgrenChrissCost {
    double theta; // spread/permanent impact coefficient
    double eta;   // temporary impact, derived from live L2 depth
    double gamma; // volatility penalty weight

    static double calibrate_eta(double bid_depth, double ask_depth, double base_eta = 1e-5) {
        double avg_depth = 0.5 * (bid_depth + ask_depth);
        return base_eta / std::max(avg_depth, 1.0); // thinner book -> higher impact
    }

    double cost(double delta_w, double tau, double sigma) const {
        return theta * std::abs(delta_w) + eta * (delta_w * delta_w) / tau + gamma * delta_w * sigma;
    }

    // gradient wrt delta_w, fed straight into the solver's temporal coupling block
    double grad(double delta_w, double tau, double sigma) const {
        return theta * ((delta_w > 0) - (delta_w < 0)) + 2.0 * eta * delta_w / tau + gamma * sigma;
    }
};
