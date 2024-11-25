#ifndef BITMASK_HEADER_FILE_H
#define BITMASK_HEADER_FILE_H

// clang-format off
#include <mutex>
#include <queue>
#include <vector>
#include "common.h"
// clang-format on

template <typename T>
class Bitmask {
   public:
    Bitmask(int n) : n_elements_(n), current_marker_(-1) {}

    void advance() {
        this->current_marker_ += 1;
        if (this->current_marker_ <= 0) {
            auto opt = torch::TensorOptions();
            if constexpr (std::is_same<T, int8_t>::value) {
                this->tensor_ = torch::zeros(
                    {this->n_elements_}, opt.dtype(torch::kInt8)
                );
            } else if constexpr (std::is_same<T, int32_t>::value) {
                this->tensor_ = torch::zeros(
                    {this->n_elements_}, opt.dtype(torch::kInt32)
                );
            } else {
                throw std::runtime_error("incorrect bitmask type");
            }
            this->current_marker_ = 1;
        }
    }

    inline T *data_ptr() { return this->tensor_.data_ptr<T>(); }

    torch::Tensor tensor() { return this->tensor_; }

    T marker() { return this->current_marker_; }

   private:
    int n_elements_;
    T current_marker_;
    torch::Tensor tensor_;
};

template <typename T>
using BitmaskPtr = std::unique_ptr<Bitmask<T>>;

template <typename T>
class BitmaskPool {
   public:
    BitmaskPool(int n) : n_elements_(n) {}

    void put(BitmaskPtr<T> &bitmask_ptr) {
        auto _ = std::lock_guard(mutex_);
        pool_.push(std::move(bitmask_ptr));
    }

    BitmaskPtr<T> get() {
        auto _ = std::lock_guard(mutex_);
        if (pool_.size() == 0) {
            return std::make_unique<Bitmask<T>>(n_elements_);
        }
        auto ptr = std::move(pool_.front());
        pool_.pop();
        return ptr;
    }

    int n_elements() { return this->n_elements_; }

   private:
    int n_elements_;
    std::mutex mutex_;
    std::queue<BitmaskPtr<T>> pool_;
};

#endif
