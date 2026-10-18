#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <stdexcept>
#include "order_book.hpp"

struct LobsterMsg {
    double time_sec;
    int type;
    long order_id;
    double size;
    double price;
    int direction;
};

// memory maps a LOBSTER message csv and parses lines without touching iostreams
class LobsterReader {
public:
    explicit LobsterReader(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        fstat(fd_, &st);
        size_ = st.st_size;
        data_ = static_cast<const char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        madvise((void*)data_, size_, MADV_SEQUENTIAL);
    }

    ~LobsterReader() { munmap((void*)data_, size_); close(fd_); }

    std::vector<LobsterMsg> parse_all() {
        std::vector<LobsterMsg> out;
        out.reserve(size_ / 32);
        size_t i = 0;
        while (i < size_) {
            LobsterMsg m{};
            m.time_sec   = read_double(i);
            m.type       = (int)read_double(i);
            m.order_id   = (long)read_double(i);
            m.size       = read_double(i);
            m.price      = read_double(i) / 10000.0; // LOBSTER prices scaled by 1e4
            m.direction  = (int)read_double(i);
            out.push_back(m);
        }
        return out;
    }

private:
    double read_double(size_t& i) {
        while (i < size_ && (data_[i] == ',' || data_[i] == '\n')) i++;
        char buf[32]; int n = 0;
        while (i < size_ && data_[i] != ',' && data_[i] != '\n') buf[n++] = data_[i++];
        buf[n] = 0;
        return atof(buf);
    }

    int fd_;
    const char* data_;
    size_t size_;
};
