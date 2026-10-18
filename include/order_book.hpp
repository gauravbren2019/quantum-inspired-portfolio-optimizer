#pragma once
#include <map>
#include <cstdint>

class OrderBook {
public:
    void apply(int type, double price, double size, int direction) {
        if (direction == 1) {
            apply_to(bids_, type, price, size);
        } else {
            apply_to(asks_, type, price, size);
        }
    }

    double best_bid() const { return bids_.empty() ? 0.0 : bids_.rbegin()->first; }
    double best_ask() const { return asks_.empty() ? 0.0 : asks_.begin()->first; }
    double mid() const { return (best_bid() + best_ask()) * 0.5; }

    double bid_depth(int levels) const { return sum_levels(bids_.rbegin(), bids_.rend(), levels); }
    double ask_depth(int levels) const { return sum_levels(asks_.begin(), asks_.end(), levels); }

private:
    template <typename MapT>
    void apply_to(MapT& book, int type, double price, double size) {
        switch (type) {
            case 1:
                book[price] += size;
                break;
            case 2:
            case 3:
                book[price] -= size;
                if (book[price] <= 1e-9) book.erase(price);
                break;
            case 4:
            case 5:
                book[price] -= size;
                if (book[price] <= 1e-9) book.erase(price);
                break;
            default:
                break;
        }
    }

    template <typename It>
    double sum_levels(It it, It end, int levels) const {
        double total = 0;
        int n = 0;
        for (; it != end && n < levels; ++it, ++n) total += it->second;
        return total;
    }

    std::map<double, double, std::greater<double>> bids_;
    std::map<double, double> asks_;
};
