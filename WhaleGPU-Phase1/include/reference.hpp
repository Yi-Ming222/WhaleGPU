#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include "common.cuh"

inline std::vector<float> reference_add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}
// Independent sequential CPU path, double accumulator for SUM.
inline double reference_reduce(const std::vector<float>& a, Op op) {
    double value = op == Op::Sum ? 0.0 : (op == Op::Max ?
        -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
    for (float x : a) {
        if (op == Op::Sum) value += static_cast<double>(x);
        else if (op == Op::Max) value = std::max(value, static_cast<double>(x));
        else value = std::min(value, static_cast<double>(x));
    }
    return value;
}
