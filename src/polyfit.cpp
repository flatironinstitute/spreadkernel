#include <polyfit.h>

#include <doctest/doctest.h>
#include <xsimd/xsimd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace spreadkernel::polyfit {
std::array<std::array<evaluator<float>, SPREADKERNEL_MAX_WIDTH + 1>, SPREADKERNEL_MAX_ORDER + 1> float_evaluators{
    {nullptr}};
std::array<std::array<evaluator<double>, SPREADKERNEL_MAX_WIDTH + 1>, SPREADKERNEL_MAX_ORDER + 1> double_evaluators{
    {nullptr}};

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
    fit(f, lb, ub, order, data, coeffs.data(), Real(0.0));
    return coeffs;
}

template <typename Real>
Real eval_multi(const std::vector<Real> &coeffs, Real x, Real h, int width, int order) {
    const Real half_h = 0.5 * h;
    const int n       = coeffs.size() / order;
    const Real lb     = -half_h * (width - 1);
    const int i       = std::min(n - 1, int((x + half_h - lb) / h));
    const Real x_i    = lb + h * i;
    assert(std::abs(x - x_i) - half_h <= std::numeric_limits<Real>::epsilon());
    return eval(coeffs.data() + i * order, order, x - x_i);
}

template <typename Real>
void fit_multi(kernel_func f, Real h, int width, int order, const void *data, Real *coeffs) {
    const Real half_h = 0.5 * h;
    const Real lb     = -half_h * (width - 1);
    for (int i = 0; i < width; i++) {
        const Real offset = lb + h * i;
        fit(f, -half_h, half_h, order, data, coeffs + i * order, offset);
    }
}

template <typename Real>
std::tuple<Real, Real> get_errors_for_auto(kernel_func f, const std::vector<Real> &sample_points,
                                           const std::vector<Real> &actual_vals, int order, const void *data,
                                           const std::vector<Real> &coeffs, Real h, int width, const int n_samples,
                                           bool use_rel_error = false) {
    Real max_error = 0.0;
    Real avg_error = 0.0;

    for (int i = 0; i < n_samples; i++) {
        Real x        = sample_points[i];
        Real y        = eval_multi(coeffs, x, h, width, order);
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
std::vector<Real> fit_multi_auto(kernel_func f, Real h, int width, const void *data, double error_tol, int min_order,
                                 int max_order, int n_samples) {
    const Real lb       = -0.5 * h * (width - 1);
    const Real ub       = -lb;
    std::vector<Real> x = linspaced(n_samples, lb - Real(0.5 * h), ub + Real(0.5 * h));
    std::vector<Real> y(n_samples);
    for (int i = 0; i < n_samples; i++)
        y[i] = f(x[i], data);

    for (int order = min_order; order < max_order; order++) {
        std::vector<Real> coeffs(order * width);
        fit_multi(f, h, width, order, data, coeffs.data());

        auto [avg_rel_error, max_rel_error] =
            get_errors_for_auto(f, x, y, order, data, coeffs, h, width, n_samples, true);
        if (avg_rel_error < error_tol) return coeffs;
    }

    return std::vector<Real>();
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
Polyfit<Real>::Polyfit(kernel_func f, const void *data, Real h, int n_spread, double tol, int min_order, int max_order,
                       int n_samples)
    : lb(-0.5 * h * (n_spread - 1)), ub(0.5 * h * (n_spread - 1)), width(n_spread), h(h), half_h(0.5 * h),
      inv_h(1.0 / h) {

    coeffs_vec = fit_multi_auto(f, h, width, data, tol, min_order, max_order, n_samples);
    order      = coeffs_vec.size() / width;
    assert(order * width == coeffs_vec.size());

    if constexpr (std::is_same_v<float, Real>)
        vec_eval = float_evaluators[width][order];
    else if constexpr (std::is_same_v<double, Real>)
        vec_eval = double_evaluators[width][order];

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < order; ++j)
            coeffs_arr[j][i] = coeffs_vec[i * order + j];
}

template <typename Real>
Real Polyfit<Real>::eval(Real x) const {
    const int i    = std::min(width - 1, int(inv_h * (x + half_h - lb)));
    const Real x_i = lb + h * i;
    return spreadkernel::polyfit::eval(coeffs_vec.data() + i * order, order, x - x_i);
}

template <typename Real>
void Polyfit<Real>::eval(Real x, Real *SPREADKERNEL_RESTRICT y) const {
    vec_eval(y, coeffs_arr, x);
}

template <typename Real, int Width = SPREADKERNEL_MAX_WIDTH, int Order = SPREADKERNEL_MAX_ORDER>
bool fill_evaluators(auto &eval) {
    eval[Width][Order] = eval_kernel_vec_horner<Width, Order, PaddedSIMD<Real, Width>>;
    if constexpr (Width == SPREADKERNEL_MIN_WIDTH && Order == SPREADKERNEL_MIN_ORDER)
        return true;
    else if constexpr (Order == SPREADKERNEL_MIN_ORDER)
        return fill_evaluators<Real, Width - 1, SPREADKERNEL_MAX_ORDER>(eval);
    else
        return fill_evaluators<Real, Width, Order - 1>(eval);
}

template class Polyfit<float>;
template class Polyfit<double>;

bool ffilled = fill_evaluators<float>(float_evaluators);
bool dfilled = fill_evaluators<double>(double_evaluators);

} // namespace spreadkernel::polyfit

TEST_CASE_TEMPLATE("SPREADKERNEL Polyfit vector eval", Real, float, double) {
    using namespace spreadkernel::polyfit;
    const Real tol  = std::is_same_v<Real, float> ? 1E-4 : 1E-8;
    const int width = 7;
    const Real h    = 1.0;
    const Real lb   = -0.5 * h * (width - 1);

    kernel_func f = [](double x, const void *data) {
        return exp(-x * x);
    };

    Polyfit<Real> polyfit(f, nullptr, h, width, tol, SPREADKERNEL_MIN_WIDTH, SPREADKERNEL_MAX_WIDTH, 100);
    REQUIRE(polyfit.order);

    alignas(64) std::array<Real, SPREADKERNEL_MAX_ORDER> res;
    alignas(64) std::array<Real, SPREADKERNEL_MAX_ORDER> res_vec;
    const Real dx = h * 0.3;
    polyfit(dx, res_vec.data());
    for (int i = 0; i < width; ++i) {
        const Real x = lb + i * h + dx;
        res[i]       = polyfit(x);
    }

    for (int i = 0; i < width; ++i)
        CHECK(std::abs(res[i] - res_vec[i]) < std::numeric_limits<Real>::epsilon());
}

TEST_CASE_TEMPLATE("SPREADKERNEL fits/evals", Real, float, double) {
    using namespace spreadkernel::polyfit;
    const Real lb         = -0.7;
    const Real ub         = 0.7;
    const int order       = SPREADKERNEL_MAX_ORDER;
    const int n_samples   = 100;
    const int max_order   = SPREADKERNEL_MAX_ORDER;
    const int min_order   = SPREADKERNEL_MIN_ORDER;
    const int width       = 7;
    const Real h          = (ub - lb) / width;
    const int multi_order = 7;
    const Real tol        = std::is_same_v<Real, float> ? 1E-3 : 1E-8;
    const void *data      = nullptr;

    kernel_func f = [](double x, const void *data) {
        return sin(pow(x, 2) * cos(x));
    };

    std::vector<Real> coeffs = fit(f, lb, ub, order, data);
    std::vector<Real> x      = linspaced(n_samples, lb, ub - std::numeric_limits<Real>::epsilon());
    std::vector<Real> y(n_samples);

    for (int i = 0; i < n_samples; i++)
        y[i] = eval(coeffs.data(), order, x[i]);

    std::vector<Real> yraw(n_samples);
    for (int i = 0; i < n_samples; i++)
        yraw[i] = f(x[i], data);

    CHECK(abs_error(y, yraw) < tol);

    std::vector<Real> coeffs_multi(multi_order * width);
    fit_multi(f, h, width, multi_order, data, coeffs_multi.data());
    alignas(64) std::array<std::array<Real, SPREADKERNEL_MAX_WIDTH>, SPREADKERNEL_MAX_ORDER> coeffs_arr = {{0.0}};
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < multi_order; ++j)
            coeffs_arr[j][i] = coeffs_multi[i * multi_order + j];

    alignas(64) std::array<Real, SPREADKERNEL_MAX_ORDER> res_vec{0.0};
    alignas(64) std::array<Real, SPREADKERNEL_MAX_ORDER> res_scalar{0.0};
    const Real dx = 0.1;
    eval_kernel_vec_horner<width, multi_order, PaddedSIMD<Real, width>>(res_vec.data(), coeffs_arr, dx);
    for (int i = 0; i < width; ++i)
        res_scalar[i] = eval(coeffs_multi.data() + i * multi_order, multi_order, Real(0.1));

    for (int i = 0; i < width; ++i)
        CHECK(std::abs(1.0 - res_vec[i] / res_scalar[i]) <= std::numeric_limits<Real>::epsilon());

    {
        std::vector<Real> y_multi(n_samples);
        for (int i = 0; i < n_samples; i++)
            y_multi[i] = eval_multi(coeffs_multi, x[i], h, width, multi_order);
        CHECK(abs_error(y_multi, yraw) < tol);
    }

    auto coeffs_auto = fit_multi_auto(f, h, width, data, tol, min_order, max_order, n_samples);
    auto auto_order  = coeffs_auto.size() / width;
    REQUIRE(auto_order);

    std::vector<Real> y_auto(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_auto[i] = eval_multi(coeffs_auto, x[i], h, width, auto_order);
    CHECK(abs_error(y_auto, yraw) < tol);

    Polyfit polyfit(f, data, h, width, tol, min_order, max_order, n_samples);
    std::vector<Real> y_polyfit(n_samples);
    for (int i = 0; i < n_samples; i++)
        y_polyfit[i] = polyfit(x[i]);

    CHECK(abs_error(y_polyfit, yraw) < tol);
}
