#ifndef POLYFIT_H
#define POLYFIT_H

#include <spreadkernel_defs.h>

#include <array>
#include <cstdint>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace { // anonymous namespace for internal structs equivalent to declaring everything
            // static
template <unsigned cap>
struct reverse_index {
    static constexpr unsigned get(unsigned index, const unsigned size) {
        return index < cap ? (cap - 1 - index) : index;
    }
};
template <unsigned cap>
struct shuffle_index {
    static constexpr unsigned get(unsigned index, const unsigned size) {
        return index < cap ? (cap - 1 - index) : size + size + cap - 1 - index;
    }
};

template <class T, uint8_t N = 1>
constexpr uint8_t min_simd_width() {
    // finds the smallest simd width that can handle N elements
    // simd size is batch size the SIMD width in xsimd terminology
    if constexpr (std::is_void_v<xsimd::make_sized_batch_t<T, N>>) {
        return min_simd_width<T, N * 2>();
    } else {
        return N;
    }
};

template <class T, uint8_t N>
constexpr auto find_optimal_simd_width() {
    // finds the smallest simd width that minimizes the number of iterations
    // NOTE: might be suboptimal for some cases 2^N+1 for example
    // in the future we might want to implement a more sophisticated algorithm
    uint8_t optimal_simd_width = min_simd_width<T>();
    uint8_t min_iterations     = (N + optimal_simd_width - 1) / optimal_simd_width;
    for (uint8_t simd_width = optimal_simd_width; simd_width <= xsimd::batch<T, xsimd::best_arch>::size;
         simd_width *= 2) {
        uint8_t iterations = (N + simd_width - 1) / simd_width;
        if (iterations < min_iterations) {
            min_iterations     = iterations;
            optimal_simd_width = simd_width;
        }
    }
    return optimal_simd_width;
}

template <class T, uint8_t N>
constexpr auto GetPaddedSIMDWidth() {
    // helper function to get the SIMD width with padding for the given number of elements
    // that minimizes the number of iterations
    return xsimd::make_sized_batch<T, find_optimal_simd_width<T, N>()>::type::size;
}

template <class T, uint8_t N>
using PaddedSIMD = typename xsimd::make_sized_batch<T, GetPaddedSIMDWidth<T, N>()>::type;

} // namespace

namespace spreadkernel::polyfit {
using kernel_func = double (*)(double, const void *);

template <typename Real>
using evaluator = void (*)(Real *SPREADKERNEL_RESTRICT,
                           const std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> &,
                           Real);

template <typename Real>
class Polyfit {
  public:
    Polyfit(kernel_func f, const void *data, Real h, int n_spread, double tol, int min_order, int max_order,
            int n_samples);
    Polyfit() = default;

    Real eval(Real x) const;
    void eval(Real x, Real *SPREADKERNEL_RESTRICT y) const;
    void operator()(Real x, Real *SPREADKERNEL_RESTRICT y) const { eval(x, y); }
    Real operator()(Real x) const { return eval(x); }

    Real lb{0.0}, ub{0.0};
    int width{0};
    int order{0};
    Real h{0.0};
    Real half_h{0.0};
    Real inv_h{0.0};
    std::vector<Real> coeffs_vec{};
    evaluator<Real> vec_eval{nullptr};
    alignas(64) std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> coeffs_arr{{0.0}};
};

} // namespace spreadkernel::polyfit

#endif
