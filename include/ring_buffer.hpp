#pragma once
#include <atomic>
#include <array>
#include <cstddef>

// single producer single consumer ring buffer, cache line padded
// the NIC thread writes, the solver thread reads, neither ever waits on the other
template <typename T, size_t Capacity>
class RingBuffer {
public:
    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) return false; // full, drop oldest signal upstream
        buf_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false; // empty
        out = buf_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

private:
    static_assert((Capacity & (Capacity - 1)) == 0, "capacity must be power of two");
    static constexpr size_t mask_ = Capacity - 1;
    std::array<T, Capacity> buf_{};
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

struct TickEvent {
    uint64_t ts_ns;
    uint16_t symbol_id;
    double price;
    double size;
    int8_t side; // +1 bid, -1 ask
};
