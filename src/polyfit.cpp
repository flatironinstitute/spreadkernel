#include <polyfit.h>
#include <spreadkernel.h>

#include <doctest/doctest.h>
#include <type_traits>
#include <xsimd/xsimd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace { // anonymous namespace for internal structs equivalent to declaring everything
            // static
struct zip_low;
struct zip_hi;
template <unsigned cap>
struct reverse_index;
template <unsigned cap>
struct shuffle_index;
struct select_even;
struct select_odd;
// forward declaration to clean up the code and be able to use this everywhere in the file
template <class T, uint8_t N, uint8_t K = N>
static constexpr auto BestSIMDHelper();

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
template <class T>
uint8_t get_padding(uint8_t ns);
template <class T, uint8_t ns>
constexpr auto get_padding();
template <class T, uint8_t N>
using BestSIMD = typename decltype(BestSIMDHelper<T, N, xsimd::batch<T>::size>())::type;
// template <class arch_t>
// constexpr auto zip_low_index = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, zip_low>();
// template <class arch_t>
// constexpr auto zip_hi_index = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, zip_hi>();
// template <class arch_t>
// constexpr auto select_even_mask = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t,
// select_even>(); template <class arch_t> constexpr auto select_odd_mask =
// xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<FLT>, arch_t, select_odd>(); template <typename T>
// SPREADKERNEL_ALWAYS_INLINE auto xsimd_to_array(const T &vec) noexcept;
} // namespace

namespace spreadkernel::polyfit {

template <typename Real>
std::vector<Real> vandermonde_inverse(int order, Real *x) {
    // https://github.com/yveschris/possibly-the-fastest-analytical-inverse-of-vandermonde-matrices
    // No license was specified, though the algorithm is generally known
    std::vector<Real> p(order + 1);
    std::vector<Real> C(order);

    C[0]         = 1;
    p[order - 1] = -x[0];
    p[order]     = 1;
    for (int i = 1; i < order; i++) {
        auto pt = p;
        for (int j = 0; j < i + 1; j++) {
            p[order - i + j - 1] = pt[order - i + j - 1] - x[i] * pt[order - i + j];
        }

        auto Cp = C;
        for (int j = 0; j < i; j++) {
            C[j] = Cp[j] / (x[j] - x[i]);
            C[i] -= C[j];
        }
    }

    std::vector<Real> Vinv(order * order);
    std::vector<Real> c(order);
    for (int i = 0; i < order; i++) {
        c[order - 1] = 1;
        for (int j = order - 2; j >= 0; j--)
            c[j] = p[j + 1] + x[i] * c[j + 1];

        for (int j = 0; j < order; j++)
            Vinv[j * order + i] = c[j] * C[i];
    }

    return Vinv;
}

template <typename Real>
std::vector<Real> linspaced(int n, Real lb, Real ub) {
    std::vector<Real> x(n);
    for (int i = 0; i < n; i++)
        x[i] = lb + (ub - lb) * i / (n - 1);

    return x;
}

template <typename Real>
void fit(kernel_func f, Real lb, Real ub, int order, const void *data, Real *coeffs, Real offset) {
    std::vector<Real> x = linspaced(order, lb, ub);

    std::vector<Real> y(order);
    for (int i = 0; i < order; i++)
        y[i] = f(x[i] + offset, data);

    std::vector<Real> Vinv = vandermonde_inverse(order, x.data());
    for (int i = 0; i < order; i++) {
        coeffs[order - i - 1] = 0;
        for (int j = 0; j < order; j++)
            coeffs[order - i - 1] += Vinv[i * order + j] * y[j];
    }
}

template <typename Real>
inline Real eval(const Real *coeffs, int order, Real x) {
    Real y = coeffs[0];
    for (auto j = 1; j < order; ++j)
        y = std::fma(y, x, coeffs[j]);

    return y;
}

template <typename Real>
std::vector<Real> fit(kernel_func f, Real lb, Real ub, int order, const void *data) {
    std::vector<Real> coeffs(order);
    fit(f, lb, ub, order, data, coeffs.data(), 0.0);
    return coeffs;
}

template <typename Real>
Real eval_multi(const std::vector<Real> &coeffs, Real x, Real lb, Real ub, int order, Real h) {
    const Real half_h = 0.5 * h;
    const int n       = coeffs.size() / order;
    const int i       = std::min(n - 1, int((x - lb) / h));
    const Real x_i    = lb + half_h * (2 * i + 1);
    return eval(coeffs.data() + i * order, order, x - x_i);
}

template <typename Real>
void fit_multi(kernel_func f, Real lb, Real ub, int order, const void *data, Real *coeffs, int n) {
    const Real half_h = (ub - lb) / (2 * n);

    for (int i = 0; i < n; i++) {
        const Real offset = lb + half_h * (2 * i + 1);
        fit(f, -half_h, half_h, order, data, coeffs + i * order, offset);
    }
}

template <typename Real>
std::tuple<Real, Real> get_errors_for_auto(kernel_func f, const std::vector<Real> &sample_points,
                                           const std::vector<Real> &actual_vals, int order, const void *data,
                                           const std::vector<Real> &coeffs, Real lb, Real ub, Real h,
                                           const int n_samples, bool use_rel_error = false) {
    Real max_error = 0.0;
    Real avg_error = 0.0;

    for (int i = 0; i < n_samples; i++) {
        Real x        = sample_points[i];
        Real y        = eval_multi(coeffs, x, lb, ub, order, h);
        Real actual_y = actual_vals[i];
        Real error    = use_rel_error ? std::abs(1.0 - y / actual_y) : std::abs(y - actual_y);
        avg_error += error;
        if (error > max_error) {
            max_error = error;
        }
    }
    avg_error /= n_samples;

    return {avg_error, max_error};
}

template <typename Real>
std::vector<Real> fit_multi_auto(kernel_func f, Real lb, Real ub, const void *data, int n_poly, double error_tol,
                                 int min_order, int max_order, int n_samples) {
    Real ub_safe       = ub - std::numeric_limits<Real>::epsilon();
    uint64_t n_epsilon = 1;
    while (ub_safe == ub) {
        ub_safe -= n_epsilon * std::numeric_limits<Real>::epsilon();
        n_epsilon *= 2;
    }

    std::vector<Real> x = linspaced(n_samples, lb, ub_safe);
    std::vector<Real> y(n_samples);
    for (int i = 0; i < n_samples; i++)
        y[i] = f(x[i], data);

    for (int order = min_order; order < max_order; order++) {
        std::vector<Real> coeffs(order * n_poly);
        fit_multi(f, lb, ub, order, data, coeffs.data(), n_poly);

        auto [avg_rel_error, max_rel_error] =
            get_errors_for_auto(f, x, y, order, data, coeffs, lb, ub, (ub - lb) / n_poly, n_samples);
        if (avg_rel_error < error_tol) return coeffs;
    }

    return std::vector<Real>();
}

template <typename Real>
void eval_grid(const std::vector<Real> &coeffs, Real x, Real lb, Real ub, int order, int n, Real *output) {
    const Real h      = (ub - lb) / n;
    const Real half_h = 0.5 * h;
    const int i       = (x - lb) / h;
    const Real dx     = x - (lb + half_h * (2 * i + 1));
    for (int j = 0; j < n; j++)
        output[j] = eval(coeffs.data() + j * order, order, dx);
}

template <typename T, std::size_t N, std::size_t M, std::size_t PaddedM>
constexpr std::array<std::array<T, PaddedM>, N> pad_2D_array_with_zeros(
    const std::array<std::array<T, M>, N> &input) noexcept {
    constexpr auto pad_with_zeros = [](const auto &input) constexpr noexcept {
        std::array<T, PaddedM> padded{0};
        for (auto i = 0; i < input.size(); ++i) {
            padded[i] = input[i];
        }
        return padded;
    };
    std::array<std::array<T, PaddedM>, N> output{};
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = pad_with_zeros(input[i]);
    }
    return output;
}

template <uint8_t width, uint8_t poly_order, class simd_type, typename Real = typename simd_type::value_type>
void eval_kernel_vec_horner(
    Real *SPREADKERNEL_RESTRICT ker,
    const std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> &padded_coeffs,
    Real x) noexcept {
    static_assert(std::is_same_v<Real, typename simd_type::value_type>,
                  "simd_type and Real template arguments must use the same underlying floating point format");
    using arch_t                      = typename simd_type::arch_type;
    static constexpr auto alignment   = arch_t::alignment();
    static constexpr auto simd_size   = simd_type::size;
    static constexpr auto padded_ns   = (width + simd_size - 1) & ~(simd_size - 1);
    static constexpr auto use_ker_sym = (simd_size < width);

    // use kernel symmetry trick if w > simd_size
    if constexpr (use_ker_sym) {
        static constexpr uint8_t tail          = width % simd_size;
        static constexpr uint8_t if_odd_degree = ((poly_order + 1) % 2);
        static constexpr uint8_t offset_start  = tail ? width - tail : width - simd_size;
        static constexpr uint8_t end_idx       = (width + (tail > 0)) / 2;
        const simd_type zv{x};
        const auto z2v = zv * zv;

        // some xsimd constant for shuffle or inverse
        static constexpr auto shuffle_batch = []() constexpr noexcept {
            if constexpr (tail) {
                return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<Real>, arch_t, shuffle_index<tail>>();
            } else {
                return xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<Real>, arch_t,
                                                  reverse_index<simd_size>>();
            }
        }();

        // process simd vecs
        simd_type k_prev, k_sym{0};
        for (uint8_t i{0}, offset = offset_start; i < end_idx; i += simd_size, offset -= simd_size) {
            auto k_odd = [i, &padded_coeffs]() constexpr noexcept {
                if constexpr (if_odd_degree) {
                    return simd_type::load_aligned(padded_coeffs[0].data() + i);
                } else {
                    return simd_type{0};
                }
            }();
            auto k_even = simd_type::load_aligned(padded_coeffs[if_odd_degree].data() + i);
            for (uint8_t j{1 + if_odd_degree}; j < poly_order; j += 2) {
                const auto cji_odd  = simd_type::load_aligned(padded_coeffs[j].data() + i);
                const auto cji_even = simd_type::load_aligned(padded_coeffs[j + 1].data() + i);
                k_odd               = xsimd::fma(k_odd, z2v, cji_odd);
                k_even              = xsimd::fma(k_even, z2v, cji_even);
            }
            // left part
            xsimd::fma(k_odd, zv, k_even).store_aligned(ker + i);
            // right part symmetric to the left part
            if (offset >= end_idx) {
                if constexpr (tail) {
                    // to use aligned store, we need shuffle the previous k_sym and current k_sym
                    k_prev = k_sym;
                    k_sym  = xsimd::fnma(k_odd, zv, k_even);
                    xsimd::shuffle(k_sym, k_prev, shuffle_batch).store_aligned(ker + offset);
                } else {
                    xsimd::swizzle(xsimd::fnma(k_odd, zv, k_even), shuffle_batch).store_aligned(ker + offset);
                }
            }
        }
    } else {
        const simd_type zv(x);
        for (uint8_t i = 0; i < width; i += simd_size) {
            auto k = simd_type::load_aligned(padded_coeffs[0].data() + i);
            for (uint8_t j = 1; j < poly_order; ++j) {
                const auto cji = simd_type::load_aligned(padded_coeffs[j].data() + i);
                k              = xsimd::fma(k, zv, cji);
            }
            k.store_aligned(ker + i);
        }
    }
}

double abs_error(const auto &y1, const auto &y2) {
    double error = 0.0;
    for (int i = 0; i < y1.size(); i++)
        error += std::abs(y1[i] - y2[i]);

    return error / y1.size();
}

template <typename Real>
Polyfit<Real>::Polyfit(kernel_func f, const void *data, Real lb, Real ub, int n_poly, double tol, int min_order,
                       int max_order, int n_samples)
    : lb(lb), ub(ub), n_poly(n_poly), h((ub - lb) / n_poly) {
    inv_h  = 1.0 / h;
    half_h = 0.5 * h;
    coeffs = fit_multi_auto(f, lb, ub, data, n_poly, tol, min_order, max_order, n_samples);
    order  = coeffs.size() / n_poly;
}

template <typename Real>
Real Polyfit<Real>::eval(Real x) const {
    const int i    = std::min(n_poly - 1, int(inv_h * (x - lb)));
    const Real x_i = lb + half_h * (2 * i + 1);
    return spreadkernel::polyfit::eval(coeffs.data() + i * order, order, x - x_i);
}

template class Polyfit<float>;
template class Polyfit<double>;

} // namespace spreadkernel::polyfit

TEST_CASE("SPREADKERNEL fits/evals") {
    using namespace spreadkernel::polyfit;
    const double lb       = -0.7;
    const double ub       = 0.8;
    const int order       = 16;
    const int n_samples   = 100;
    const int max_order   = SPREADKERNEL_MAX_ORDER;
    const int min_order   = SPREADKERNEL_MIN_ORDER;
    const int width       = 7;
    const double h        = (ub - lb) / width;
    const int multi_order = 8;
    const double tol      = 1E-8;

    void *data    = NULL;
    kernel_func f = [](double x, const void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    std::vector<double> coeffs = fit(f, lb, ub, order, data);
    std::vector<double> x      = linspaced(n_samples, lb, ub - std::numeric_limits<double>::epsilon());
    std::vector<double> y(n_samples);

    for (int i = 0; i < n_samples; i++)
        y[i] = eval(coeffs.data(), order, x[i]);

    std::vector<double> yraw(n_samples);
    for (int i = 0; i < n_samples; i++)
        yraw[i] = f(x[i], data);

    CHECK(abs_error(y, yraw) < tol);

    std::vector<double> coeffs_multi(multi_order * width);
    fit_multi(f, lb, ub, multi_order, data, coeffs_multi.data(), width);
    alignas(64) std::array<std::array<double, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> coeffs_arr = {{0.0}};
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < multi_order; ++j)
            coeffs_arr[j][i] = coeffs_multi[i * multi_order + j];

    alignas(32) std::array<double, width> res_vec{0.0};
    alignas(32) std::array<double, width> res_scalar{0.0};
    eval_kernel_vec_horner<width, multi_order, PaddedSIMD<double, width>>(res_vec.data(), coeffs_arr, 0.1);
    for (int i = 0; i < width; ++i)
        res_scalar[i] = eval(coeffs_multi.data() + i * multi_order, multi_order, 0.1);

    CHECK(res_vec == res_scalar);

    {
        std::vector<double> y_multi(n_samples);
        for (int i = 0; i < n_samples; i++)
            y_multi[i] = eval_multi(coeffs_multi, x[i], lb, ub, multi_order, h);
        CHECK(abs_error(y_multi, yraw) < tol);
    }

    std::vector<double> y_grid(width);
    std::vector<double> x_grid_multi = linspaced(width, lb + 0.75 * h, ub - 0.25 * h);
    std::vector<double> y_grid_multi(width);
    eval_grid(coeffs_multi, lb + 1.75 * h, lb, ub, multi_order, width, y_grid.data());
    for (int i = 0; i < width; i++)
        y_grid_multi[i] = f(x_grid_multi[i], data);
    CHECK(abs_error(y_grid, y_grid_multi) < tol);

    auto coeffs_auto = fit_multi_auto(f, lb, ub, data, width, tol, min_order, max_order, n_samples);
    auto auto_order  = coeffs_auto.size() / width;

    std::vector<double> y_auto(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_auto[i] = eval_multi(coeffs_auto, x[i], lb, ub, auto_order, h);
    CHECK(abs_error(y_auto, yraw) < tol);

    Polyfit polyfit(f, data, lb, ub, width, tol, min_order, max_order, n_samples);
    std::vector<double> y_polyfit(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_polyfit[i] = polyfit(x[i]);

    CHECK(abs_error(y_polyfit, yraw) < tol);
}
