#ifndef BITMASK_HEADER_FILE_H
#define BITMASK_HEADER_FILE_H

// clang-format off
#include <mutex>
#include <queue>
#include <vector>
#include "common.h"
// clang-format on

// clang-format off
template <typename T>
class Bitmask {
   public:
    Bitmask(
        int n, int bsz,
        torch::Device device
    ) : n_elements_(n),
        batch_size_(bsz),
        current_marker_(-1),
        device_(device) {}

    void advance() {
        this->current_marker_ += 1;
        if (this->current_marker_ <= 0) {
            auto opt = torch::TensorOptions();
            if constexpr (std::is_same<T, int8_t>::value) {
                this->tensor_ = torch::zeros(
                    {this->batch_size_, this->n_elements_},
                    opt.dtype(torch::kInt8).device(this->device_)
                );
            } else if constexpr (std::is_same<T, int32_t>::value) {
                this->tensor_ = torch::zeros(
                    {this->batch_size_, this->n_elements_},
                    opt.dtype(torch::kInt32).device(this->device_)
                );
            } else {
                throw std::runtime_error("incorrect bitmask type");
            }
            this->current_marker_ = 1;
        }
    }

    inline T *data_ptr() {
        return this->tensor_.data_ptr<T>();
    }

    torch::Tensor tensor() { return this->tensor_; }

    T marker() { return this->current_marker_; }

   private:
    int n_elements_;
    int batch_size_;
    T current_marker_;
    torch::Tensor tensor_;
    torch::Device device_;
};

template <typename T>
using BitmaskPtr = std::unique_ptr<Bitmask<T>>;

template <typename T>
class BitmaskPool {
   public:
    BitmaskPool(
        int n, int bsz = 1,
        torch::Device device = torch::Device(torch::kCPU)
    ) : n_elements_(n), batch_size_(bsz), device_(device) {}

    void put(BitmaskPtr<T> &bitmask_ptr) {
        auto _ = std::lock_guard(mutex_);
        pool_.push(std::move(bitmask_ptr));
    }

    BitmaskPtr<T> get() {
        auto _ = std::lock_guard(mutex_);
        if (pool_.size() == 0) {
            return std::make_unique<Bitmask<T>>(
                n_elements_, batch_size_, device_
            );
        }
        auto ptr = std::move(pool_.front());
        pool_.pop();
        return ptr;
    }

    int n_elements() {
        return this->n_elements_;
    }

    int batch_size() {
        return this->batch_size_;
    }

   private:
    int n_elements_;
    int batch_size_;
    std::mutex mutex_;
    torch::Device device_;
    std::queue<BitmaskPtr<T>> pool_;
};
// clang-format on

#endif
