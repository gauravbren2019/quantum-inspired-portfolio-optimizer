#include <iostream>
#include <chrono>
#include "lobster_reader.hpp"
#include "order_book.hpp"
#include "rmt_filter.hpp"
#include "factor_model.hpp"
#include "active_set_screener.hpp"
#include "regime_detector.hpp"
#include "ewma_calibration.hpp"
#include "admm_cardinality.hpp"
#include "cfl_autotune.hpp"
#include "dinkelbach.hpp"
#include "pgd_polish.hpp"
#include "almgren_chriss.hpp"

struct SymbolStream {
    int id;
    std::vector<LobsterMsg> msgs;
    size_t cursor = 0;
    OrderBook book;
};

static double clamp_ret(double r) {
    if (!std::isfinite(r)) return 0.0;
    if (r > 0.2) return 0.2;
    if (r < -0.2) return -0.2;
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: backtest_engine <lobster_dir_with_N_symbols>\n"; return 1; }

    const int N_UNIVERSE = 500;
    const int TOP_K = 30;
    const int BITS = 8;
    const int REBUILD_EVERY = 500;

    std::vector<SymbolStream> streams;
    for (int s = 0; s < N_UNIVERSE; s++) {
        SymbolStream ss;
        ss.id = s;
        LobsterReader reader(std::string(argv[1]) + "/sym_" + std::to_string(s) + "_message.csv");
        ss.msgs = reader.parse_all();
        streams.push_back(std::move(ss));
    }

    EWMACalibrator calibrator(N_UNIVERSE);
    RegimeDetector regime;
    ADMMCardinality admm(TOP_K);

    std::vector<double> mid_prev(N_UNIVERSE, 0.0);
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(N_UNIVERSE);

    using clk = std::chrono::high_resolution_clock;
    auto us = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };

    size_t total_events = 0;
    bool advancing = true;
    int tick_counter = 0;

    // running max book size to check for unbounded growth
    size_t max_book_entries = 0;

    while (advancing) {
        advancing = false;
        double min_t = 1e18; int min_idx = -1;
        for (auto& s : streams) {
            if (s.cursor < s.msgs.size() && s.msgs[s.cursor].time_sec < min_t) {
                min_t = s.msgs[s.cursor].time_sec;
                min_idx = s.id;
                advancing = true;
            }
        }
        if (min_idx < 0) break;

        auto& s = streams[min_idx];
        auto& m = s.msgs[s.cursor++];
        s.book.apply(m.type, m.price, m.size, m.direction);
        total_events++;

        double mid = s.book.mid();
        ret(min_idx) = 0.0;
        if (mid > 0 && mid_prev[min_idx] > 0) {
            ret(min_idx) = clamp_ret((mid - mid_prev[min_idx]) / mid_prev[min_idx]);
        }
        if (mid > 0) mid_prev[min_idx] = mid;

        tick_counter++;
        if (tick_counter % REBUILD_EVERY != 0) continue;

        auto t_start = clk::now();

        calibrator.update(ret);
        regime.update(ret.mean());
        auto t_calib = clk::now();

        std::vector<double> vols(N_UNIVERSE), mu_all(N_UNIVERSE), depths(N_UNIVERSE);
        const Eigen::MatrixXd& sigma_full = calibrator.sigma();
        for (int i = 0; i < N_UNIVERSE; i++) {
            vols[i] = std::sqrt(std::max(sigma_full(i, i), 1e-12));
            mu_all[i] = calibrator.mu()(i);
            depths[i] = streams[i].book.bid_depth(5) + streams[i].book.ask_depth(5);
        }
        auto active_idx = ActiveSetScreener::screen(mu_all, vols, depths, TOP_K);
        auto t_screen = clk::now();

        Eigen::MatrixXd sigma_active(TOP_K, TOP_K);
        Eigen::VectorXd mu_active(TOP_K);
        for (int a = 0; a < TOP_K; a++) {
            mu_active(a) = calibrator.mu()(active_idx[a]);
            for (int bIdx = 0; bIdx < TOP_K; bIdx++)
                sigma_active(a, bIdx) = sigma_full(active_idx[a], active_idx[bIdx]);
        }

        Eigen::MatrixXd sigma_clean = RMTFilter::clean(sigma_active, TOP_K, 500);
        sigma_clean = RegimeDetector::enforce_psd(sigma_clean);
        auto t_rmt = clk::now();

        FactorModel fm = FactorModel::fit(sigma_clean, std::min(5, TOP_K));
        double spectral_radius = CFLAutotune::spectral_radius(fm, TOP_K);
        auto t_factor = clk::now();

        SolverConfig cfg;
        cfg.n_assets = TOP_K;
        cfg.bits_per_asset = BITS;
        cfg.xi0 = 1.0 / std::max(spectral_radius, 1e-6);
        cfg.dt = CFLAutotune::stable_dt(cfg.delta0, cfg.xi0, spectral_radius);
        cfg.steps = 150;
        cfg.noise_sigma0 = regime.is_stressed() ? 0.15 : 0.05;

        Eigen::VectorXd admm_bias = admm.dual_bias(TOP_K);
        DinkelbachSolver dink(cfg);
        auto w_raw = dink.maximize_sharpe(mu_active, fm, admm_bias);
        auto t_dink = clk::now();

        Eigen::VectorXd w(TOP_K);
        for (int a = 0; a < TOP_K; a++) w(a) = w_raw[a];
        Eigen::VectorXd w_polished = PGDPolish::polish(w, mu_active, fm, 1.0);
        auto t_pgd = clk::now();

        int active_count = 0;
        for (int a = 0; a < TOP_K; a++) if (std::abs(w_polished(a)) > 1e-3) active_count++;
        admm.update(active_count);

        size_t cur_book_total = 0;
        for (int i = 0; i < N_UNIVERSE; i++) cur_book_total += streams[i].book.bid_depth(1000000) > 0 ? 1 : 0;
        max_book_entries = std::max(max_book_entries, cur_book_total);

        if (tick_counter % 5000 == 0) {
            std::cout << "events=" << total_events
                      << " total_us=" << us(t_start, t_pgd)
                      << " calib_us=" << us(t_start, t_calib)
                      << " screen_us=" << us(t_calib, t_screen)
                      << " rmt_us=" << us(t_screen, t_rmt)
                      << " factor_us=" << us(t_rmt, t_factor)
                      << " dinkelbach_us=" << us(t_factor, t_dink)
                      << " dink_iters=" << dink.iters_used << " pgd_us=" << us(t_dink, t_pgd)
                      << " active_names=" << active_count
                      << "\n";
        }
    }

    std::cout << "backtest complete, total historical events processed=" << total_events << "\n";
    return 0;
}
