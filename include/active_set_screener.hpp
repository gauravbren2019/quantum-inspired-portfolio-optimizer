#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

// cuts a 500 name universe to the top-K before any heavy math runs
// ranks by liquidity-adjusted momentum, a cheap O(N log N) screen, no dense step-2 penalty blowup
struct ActiveSetScreener {
    static std::vector<int> screen(const std::vector<double>& mu,
                                    const std::vector<double>& vol,
                                    const std::vector<double>& depth,
                                    int top_k) {
        int N = mu.size();
        std::vector<double> score(N);
        for (int i = 0; i < N; i++)
            score[i] = (vol[i] > 1e-9) ? (mu[i] / vol[i]) * std::log(1.0 + depth[i]) : 0.0;

        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min(top_k, N), idx.end(),
            [&](int a, int b) { return std::abs(score[a]) > std::abs(score[b]); });
        idx.resize(std::min(top_k, N));
        return idx;
    }
};
